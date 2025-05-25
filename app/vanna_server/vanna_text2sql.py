import os
import shutil

from chromadb import EmbeddingFunction, Documents, Embeddings

from rewrite_ask import ask
from siliconflow_api import SiliconflowEmbedding
from custom_chat import CustomChat
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import plotly.io as pio

load_dotenv()
# 设置显示后端为浏览器
pio.renderers.default = 'browser'


class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A embeddingFunction that to generate embeddings which can use in chromadb.
    """

    def __init__(self, config=None):
        if config is None or "api_key" not in config:
            raise ValueError("Missing 'api_key' in config")

        self.api_key = config["api_key"]
        self.model = config.get("model", "BAAI/bge-m3")

        try:
            self.client = config["embedding_client"](api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Error initializing client: {e}")

    def __call__(self, input: Documents) -> Embeddings:
        # Replace newlines, which can negatively affect performance.
        input = [t.replace("\n", " ") for t in input]
        all_embeddings = []
        print(f"Generating embeddings for {len(input)} documents")

        # Iterating over each document for individual API calls
        for document in input:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=document
                )
                # print(response)
                embedding = response.data[0].embedding
                all_embeddings.append(embedding)
                # print(f"Cost required: {response.usage.total_tokens}")
            except Exception as e:
                raise ValueError(f"Error generating embedding for document: {e}")

        return all_embeddings


class VannaServer:
    def __init__(self, config):
        self.config = config
        self.vn = self._initialize_vn()

    def _initialize_vn(self):
        config = self.config
        supplier = config["supplier"]
        embedding_supplier = config["embedding_supplier"] if "embedding_supplier" in config else "SiliconFlow"
        vector_db_path = config["vector_db_path"] if "vector_db_path" in config else os.getenv("VECTOR_DB_PATH",
                                                                                               "../storage/chromadb")
        EmbeddingClass = config["EmbeddingClass"] if "EmbeddingClass" in config else SiliconflowEmbedding
        ChatClass = config["ChatClass"] if "ChatClass" in config else CustomChat
        host = config["host"] if "host" in config else os.getenv("DB_HOST", "localhost")
        dbname = config["db_name"] if "db_name" in config else os.getenv("DB_NAME", "dify_data")
        user = config["user"] if "user" in config else os.getenv("DB_USER", "root")
        password = config["password"] if "password" in config else os.getenv("DB_PASSWORD", "mysql")
        port = config["port"] if "port" in config else int(os.getenv("DB_PORT", 3306))

        os.makedirs(vector_db_path, exist_ok=True)

        config = {"api_key": os.getenv(f"{embedding_supplier}_EMBEDDING_API_KEY"),
                  "model": os.getenv(f"{embedding_supplier}_EMBEDDING_MODEL"), "embedding_client": EmbeddingClass}

        config = {"api_key": os.getenv(f"{supplier}_API_KEY"), "model": os.getenv(f"{supplier}_CHAT_MODEL"),
                  "api_base": os.getenv(f"{supplier}_API_BASE"),
                  "path": vector_db_path, "embedding_function": CustomEmbeddingFunction(config)}

        MyVanna = make_vanna_class(ChatClass=ChatClass)
        vn = MyVanna(config)

        vn.connect_to_mysql(host=host, dbname=dbname, user=user, password=password, port=port)

        self._copy_fig_html()

        return vn

    def _copy_fig_html(self):
        source_path = 'fig.html'
        target_dir = '../output/html'
        target_path = os.path.join(target_dir, 'vanna_fig.html')

        # 检查目标文件是否存在
        if os.path.exists(target_path):
            print(f"Target file {target_path} already exists. Skipping copy.")
            return

        # 确保源文件存在
        if not os.path.exists(source_path):
            print(f"Source file {source_path} does not exist.")
            return

        # 创建目标目录（如果不存在）
        os.makedirs(target_dir, exist_ok=True)

        # 复制文件
        try:
            shutil.copy(source_path, target_path)
            print(f"Successfully copied {source_path} to {target_path}")
        except Exception as e:
            print(f"Failed to copy {source_path} to {target_path}: {e}")

    def schema_train(self):
        # The information schema query may need some tweaking depending on your database. This is a good starting point.
        df_information_schema = self.vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

        # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
        plan = self.vn.get_training_plan_generic(df_information_schema)
        # print(plan)

        # If you like the plan, then uncomment this and run it to train
        self.vn.train(plan=plan)

    def vn_train(self, question="", sql="", documentation="", ddl=""):
        if question and sql:
            # 训练问答对
            self.vn.train(
                question=question,
                sql=sql
            )
        elif sql:
            # You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
            self.vn.train(sql=sql)

        if documentation:
            # Sometimes you may want to add documentation about your business terminology or definitions.
            self.vn.train(documentation=documentation)

        if ddl:
            # You can also add DDL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
            self.vn.train(ddl=ddl)

    def get_training_data(self):
        training_data = self.vn.get_training_data()
        # print(training_data)
        return training_data

    def ask(self, question, visualize=True, auto_train=True, *args, **kwargs):
        # sql = self.vn.generate_sql(question=question)
        # print("这里是生成的sql语句： ", sql)
        # df = self.vn.run_sql(sql)
        # print("\n这里是查询的数据： ", df)
        # plotly_code = self.vn.generate_plotly_code(question=question, sql=sql, df_metadata=df)
        # print("\n这里是生成的plotly代码： ", plotly_code)
        # figure = self.vn.get_plotly_figure(plotly_code, df=df)
        # # figure.show()
        # sql, df, fig = self.vn.ask(question, visualize=visualize, auto_train=auto_train)
        sql, df, fig = ask(self.vn, question, visualize=visualize, auto_train=auto_train, *args, **kwargs)
        # fig.show()
        return sql, df, fig


def make_vanna_class(ChatClass=CustomChat):
    class MyVanna(ChromaDB_VectorStore, ChatClass):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            ChatClass.__init__(self, config=config)

        def is_sql_valid(self, sql: str) -> bool:
            # Your implementation here
            return False

        def generate_query_explanation(self, sql: str):
            my_prompt = [
                self.system_message("You are a helpful assistant that will explain a SQL query"),
                self.user_message("Explain this SQL query: " + sql),
            ]

            return self.submit_prompt(prompt=my_prompt)

    return MyVanna


# 使用示例
if __name__ == '__main__':
    config = {"supplier": "GITEE"}
    server = VannaServer(config)
    # server.schema_train()
    server.ask("汇总每个类别的销售量和销售额, 并按照销售量进行降序排列")
