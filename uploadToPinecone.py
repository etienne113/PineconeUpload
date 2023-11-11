import datetime
import json
import os
import random
import tempfile

import chainlit as cl
import openai
import pinecone
from chainlit.input_widget import Tags, Select
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader, TextLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_ENV"] = os.getenv('PINECONE_ENV')

embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)
index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

welcome_message = """ Welcome to the Chainlit PDF QA demo! To get started:
  1. Upload your PDF or text file \n
  2. Ask questions about the files
  """
task1 = cl.Task(title="Select file", status=cl.TaskStatus.RUNNING)
task2 = cl.Task(title="Select department(s) ", status=cl.TaskStatus.READY)
task3 = cl.Task(title="Select the unique ID ", status=cl.TaskStatus.READY)
task4 = cl.Task(title="Storing in Pinecone Index... ", status=cl.TaskStatus.READY)
task5 = cl.Task(title="Select your filter criterium", status=cl.TaskStatus.READY)


def sort_strings_alphabetically(input_list):
    sorted_list = sorted(input_list)
    return sorted_list


def modify_metadata(id_vector, unique_id, metadata_request: str):
    not_found = False
    ids = []
    last_id = id_vector
    m_unique_id = unique_id
    data = index.query(id=id_vector, filter={'unique id': m_unique_id},
                       top_k=50,
                       include_metadata=True)
    if 'matches' in data and isinstance(data['matches'], list):
        matches = data['matches']
        for i in range(len(matches)):
            if 'id' in matches[i]:
                if i < len(matches) - 1:
                    ids.append(matches[i]['id'])
                else:
                    last_id = matches[i]['id']
                    not_found = True
    for this_id in ids:
        new_metadata = json.loads(metadata_request)
        try:
            index.update(id=this_id, set_metadata=new_metadata)
        except json.JSONDecodeError:
            print("Invalid JSON format. Please provide valid JSON data.")
    if not not_found:
        modify_metadata(last_id, m_unique_id)
    else:
        new_metadata = json.loads(metadata_request)
        try:
            index.update(id=last_id, set_metadata=new_metadata)
        except json.JSONDecodeError:
            print("Invalid JSON format. Please provide valid JSON data.")
    return


def execute(id_vector, unique_id):
    not_found = False
    ids = []
    last_id = id_vector
    m_unique_id = unique_id
    data = index.query(id=id_vector, filter={'unique id': unique_id},
                       top_k=50, include_metadata=True)
    if 'matches' in data and isinstance(data['matches'], list):
        matches = data['matches']
        for i in range(len(matches)):
            if 'id' in matches[i]:
                if i < len(matches) - 1:
                    ids.append(matches[i]['id'])
                else:
                    last_id = matches[i]['id']
                    not_found = True

    for this_id in ids:
        index.delete(id=this_id, filter={'unique id': data['matches'][0]['metadata']['unique id']})
    if not not_found:
        execute(last_id, m_unique_id)
    else:
        index.delete(id=last_id, filter={'unique id': data['matches'][0]['metadata']['unique id']})
        return


def delete_doc(data, unique_id):
    execute(data[0]['id'], unique_id)
    print('The existing file has been successful deleted ')


async def search_same_doc(unique_id: str):
    metric_dimension = 1536

    vector_for_query = [random.uniform(0.0, 1.0) for _ in range(metric_dimension)]
    result = index.query(vector=vector_for_query, top_k=1, filter={'unique id': unique_id})
    data = result['matches']
    if len(data) != 0:
        cl.user_session.set('data', data)
        return True
    else:
        return False


def process_files(files):
    processed_docs = []
    for file in files:
        if file.type == "application/pdf":
            # Handle PDF files
            Loader = PyPDFLoader
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=450)
        elif file.type == "text/plain":
            # Handle  text files
            Loader = TextLoader
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=450)
        elif file.type == "text/csv":
            Loader = CSVLoader
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=450)
        else:
            raise ValueError("Unsupported file type: " + file.type)

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(file.content)
            loader = Loader(temp_file.name)
            documents = loader.load()

            docs = splitter.split_documents(documents)
            metadata = cl.user_session.get('metadata')
            unique_id = cl.user_session.get('unique id')
            for i, doc in enumerate(docs):
                doc.metadata["source"] = f"source_{i}"
                doc.metadata['departments'] = metadata.get('departments')
                doc.metadata['unique id'] = unique_id
                doc.metadata['last modified date'] = current_datetime
                processed_docs.append(doc)
    cl.user_session.set("docs", docs)
    return processed_docs


def store_to_index(file):
    doc = process_files(file)
    if not doc:
        return None

    index_name = os.getenv('PINECONE_INDEX_NAME')
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )
    docsearch = Pinecone.from_documents(doc, embeddings, index_name=index_name)
    cl.user_session.set("doc", doc)
    return docsearch


async def task_list_exec(task_list, task_list_status, sleep_time, task, new_task, task_status):
    if new_task:
        await task_list.add_task(task)
    await cl.sleep(sleep_time)
    task_list.status = task_list_status
    task.status = task_status
    await task_list.send()


async def modify_task_and_task_list(task, new_title):
    task_list = cl.user_session.get('task_list')
    task.title = new_title
    await task_list.send()


@cl.on_chat_start
async def start():
    task_list = cl.TaskList()
    await task_list_exec(task_list, 'Running...', 0, task1, True, cl.TaskStatus.RUNNING)
    await task_list_exec(task_list, 'Adding new task...', 2, task2, True, cl.TaskStatus.READY)
    await task_list_exec(task_list, 'Adding new task...', 2, task3, True, cl.TaskStatus.READY)
    await task_list_exec(task_list, 'Adding new task...', 2, task4, True, cl.TaskStatus.READY)
    await task_list_exec(task_list, 'Adding new task...', 2, task5, True, cl.TaskStatus.READY)
    await task_list_exec(task_list, 'Running...', 0, task1, False, cl.TaskStatus.RUNNING)
    cl.user_session.set('task_list', task_list)
    await cl.Message(
        content="Welcome to this space, you can use this to store your docs into your Pinecone index and ask questions about them!").send()
    departments_selected = False
    cl.user_session.set('departments_selected', departments_selected)
    file = None

    while file is None:
        file = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf", "text/csv"],
            max_files=1,
            max_size_mb=20,
            timeout=1800,
        ).send()

    await task_list_exec(task_list, 'Done', 2, task1, False, cl.TaskStatus.DONE)
    await modify_task_and_task_list(task1, 'File selected successfully!')

    # Sending an image with the local file path
    elements = [
        cl.Image(name="image1", display="inline", size="large", path="./select_dep.png")
    ]
    await task_list_exec(task_list, 'Running...', 0, task2, False, cl.TaskStatus.RUNNING)
    temp_mes5 = cl.Message(content="Now you can choose the departments:", elements=elements)
    await temp_mes5.send()
    temp_messages.append(temp_mes5)
    input_list = ["Software engineering", "DevOps", "Project management", "Data management", "Personal management"]
    departments_list = sort_strings_alphabetically(input_list=input_list)
    setting1 = await cl.ChatSettings(
        [
            Tags(
                id="departments",
                label="Please select the department(s): ",
                initial=departments_list,
            )
        ]
    ).send()

    cl.user_session.set('files', file)


actions = [
    cl.Action(name=f"Unique ID already taken", value=""),
    cl.Action(name="Yes", value="", description="Overwrite the file!"),
    cl.Action(name="No", value="", description="Cancel the process!")
]
temp_messages = []


async def erase_alerts():
    for action in actions:
        await action.remove()
    for temp_mes in temp_messages:
        await temp_mes.remove()


async def upload_to_index():
    files = cl.user_session.get('files')
    task_list = cl.user_session.get('task_list')
    temp_mes5 = cl.Message(content=f"Your files are being uploaded...")
    await temp_mes5.send()
    temp_messages.append(temp_mes5)
    docsearch = await cl.make_async(store_to_index)(files)
    retrieverdb = docsearch.as_retriever()
    retrieverdb.search_kwargs = {'filter': {'departments': 'None'}}

    await task_list_exec(task_list, 'Done', 0, task4, False, cl.TaskStatus.DONE)
    await modify_task_and_task_list(task4, 'File uploaded succesfully!')
    await task_list_exec(task_list, 'Running...', 0, task5, False, cl.TaskStatus.RUNNING)

    input_list = ["Software engineering", "DevOps", "Project management", "Data management", "Personal management"]
    departments_list = sort_strings_alphabetically(input_list=input_list)
    filter_chooose = await cl.ChatSettings(
        [
            Select(
                id="filter for QA",
                label="You can choose : ",
                values=['None'] + departments_list,
                initial_index=0,
            )
        ]
    ).send()
    cl.user_session.set("retrieverdb", retrieverdb)
    await erase_alerts()


@cl.action_callback("Yes")
async def on_action(action):
    data = cl.user_session.get('data')
    delete_doc(data, cl.user_session.get('unique id'))
    await upload_to_index()
    await erase_alerts()
    await cl.Message(content=f"The file was succesfully uploaded in your provided Pinecone Index!" + '\n' +
                             ' You can now ask Questions to your docs').send()


@cl.action_callback("No")
async def on_action(action):
    temp_mes3 = cl.Message(
        content="The file has not been overwritten!;" + 'You can now choose another Unique ID')
    await temp_mes3.send()
    temp_messages.append(temp_mes3)
    await erase_alerts()
    await temp_mes3.send()


async def display_alert():
    tmp_actions = [actions[1], actions[2]]
    temp_mes2 = cl.Message(content="A document with the same unique ID has been found!" + '\n' +
                                   "Would you like the file to be overwritten? ", actions=tmp_actions)
    await temp_mes2.send()
    temp_messages.append(temp_mes2)


@cl.on_settings_update
async def handle_update(settings):
    docsearch = None
    task_list = cl.user_session.get('task_list')
    departments_selected = cl.user_session.get('departments_selected')
    if not departments_selected and 'departments' in settings:
        selected_options = settings['departments']
        default_value = bool
        if len(selected_options) == 0:
            metadata = {"departments": "None"}
        else:
            metadata = {"departments": selected_options}
        cl.user_session.set("metadata", metadata)
        selected = True
        cl.user_session.set('departments_selected', selected)
        alphabetic_list = [chr(ord('A') + i) for i in range(26)]
        await cl.ChatSettings(
            [
                Select(
                    id="unique id",
                    label="You can choose : ",
                    values=alphabetic_list,
                    initial_index=0,
                )
            ]
        ).send()
        await task_list_exec(task_list, 'Done', 2, task2, False, cl.TaskStatus.DONE)
        await modify_task_and_task_list(task2, 'Department(s) selected successfully!')
        await task_list_exec(task_list, 'Running...', 2, task3, False, cl.TaskStatus.RUNNING)
        select_id = "Select the unique id of your document: "

        elements = [
            cl.Image(name="image1", display="inline", size="large", path="./select_id.png")
        ]

        temp_mes4 = cl.Message(content=select_id, elements=elements)
        await temp_mes4.send()
        temp_messages.append(temp_mes4)
    elif 'unique id' in settings:
        await modify_task_and_task_list(task3, 'Unique ID selected succesfully!')
        await task_list_exec(task_list, 'Done', 2, task3, False, cl.TaskStatus.DONE)
        await task_list_exec(task_list, 'Running...', 1, task4, False, cl.TaskStatus.RUNNING)
        cl.user_session.set('unique id', settings['unique id'])
        if await search_same_doc(settings['unique id']):
            await display_alert()
        else:
            await upload_to_index()
            await cl.Message(content=f"The file was succesfully uploaded in your provided Pinecone Index!" + '\n' +
                                     ' You can now ask Questions to your docs').send()

    elif 'filter for QA' in settings:
        retrieverdb = cl.user_session.get('retrieverdb')  # type: VectorStoreRetriever
        if 'None' in settings['filter for QA']:
            pass
        else:
            retrieverdb.search_kwargs = {'filter': {'departments': settings['filter for QA']}}
        cl.user_session.set('retrieverdb', retrieverdb)
        await task_list_exec(task_list, 'Done', 0, task5, False, cl.TaskStatus.DONE)
        await modify_task_and_task_list(task5, 'Filter selected succesfully!')


@cl.on_message
async def main(message):
    retrieverdb = cl.user_session.get('retrieverdb')
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=retrieverdb
    )
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    docs = cl.user_session.get("docs")
    metadata = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadata]

    if sources:
        found_sources = []

        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)

            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {','.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
