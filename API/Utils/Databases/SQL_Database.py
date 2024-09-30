from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime

from API.DataTransferObjects.Requests.inferenceRequestDTO import InferenceRequestDTO

from settings import auto_config as cfg

Base = declarative_base()

class Operation(Base):
    __tablename__ = 'operations'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    user_name = Column(String)
    user_message = Column(String)
    system_response = Column(String)
    inference_parameters = Column(String)
    rag_parameters = Column(String)

    def __repr__(self):
        return (f"<Conversation("
                f"timestamp='{str(self.timestamp)}', "
                f"user_name='{str(self.user_name)}', "
                f"user_message='{str(self.user_message)}', "
                f"system_response='{str(self.system_response)}', "
                f"inference_parameters='{str(self.inference_parameters)}', "
                f"rag_parameters='{str(self.rag_parameters)}')>")

class Knowledge(Base):
    __tablename__ = 'knowledge'

    id = Column(Integer, primary_key=True)
    tmstmp = Column(DateTime)
    document = Column(String)

    def __repr__(self):
        return (f"<Knowledge("
                f"tmstmp='{str(self.tmstmp)}', "
                f"document='{str(self.document)}')>")

class Database:
    def __init__(self):
        self.engine = create_engine("sqlite:///" + cfg.DATABASE_FILE, echo=False)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)
        self._test_connection()
        self.max_cache_conversations = int(cfg.MAX_CACHE_CONVERSATIONS)

    # PRIVATE FUNCTIONS
    def _test_connection(self):
        try:
            con = self.engine.connect()
            con.close()
        except:
            raise ConnectionError("Cannot connect to the provided connection string.")

    # PUBLIC FUNCTIONS

    def add_operation(self, user_name: str, system_response: str, inference_data: InferenceRequestDTO):
        with self.session() as session:
            new_operation = Operation(
                timestamp=datetime.now(),
                user_name=user_name,
                user_message=inference_data.messages[-1].content,
                system_response=system_response,
                inference_parameters=inference_data.inference_parameters.__repr__(),
                rag_parameters=inference_data.rag_parameters.__repr__()
            )
            session.add(new_operation)
            session.commit()
            session.close()
        self.clear_cache()

    def clear_cache(self):
        with self.session() as session:
            operations_count = session.query(Operation).count()
            if operations_count > self.max_cache_conversations:
                rows_to_delete = operations_count - self.max_cache_conversations
                oldest_operations = session.query(Operation) \
                    .order_by(Operation.timestamp.asc()) \
                    .limit(rows_to_delete) \
                    .all()
                for operation in oldest_operations:
                    session.delete(operation)
                session.commit()
            session.close()

    def get_all_knowledge_base(self):
        with self.session() as session:
            documents = session.query(Knowledge.document).all()
            return [doc[0] for doc in documents]


DATABASE = Database()