from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from settings import auto_config as cfg
from API.DataTransferObjects.Requests.inferenceRequestDTO import InferenceRequestDTO
from datetime import datetime

Base = declarative_base()

class Operation(Base):
    __tablename__ = 'operations'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    action = Column(String)
    user_name = Column(String)
    last_message = Column(String)
    conversation = Column(String)
    temperature = Column(Float)
    tokens_count = Column(Integer)
    do_samples = Column(Boolean)
    top_k_tokens = Column(Integer)
    top_p_tokens = Column(Float)

    def __repr__(self):
        return (f"<Conversation("
                f"timestamp='{str(self.timestamp)}', "
                f"action='{str(self.action)}', "
                f"user_name='{str(self.user_name)}', "
                f"last_message='{str(self.last_message)}', "
                f"conversation='{str(self.conversation)}', "
                f"temperature='{str(self.temperature)}', "
                f"tokens_count='{str(self.tokens_count)}', "
                f"top_k_tokens='{str(self.top_k_tokens)}', "
                f"top_p_tokens='{str(self.top_p_tokens)}')>")

class CacheDatabase:
    def __init__(self):
        self.engine = create_engine(cfg.CACHE_CON_STRING, echo=False)

        self._test_connection()
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)

        self.max_cache_conversations = int(cfg.MAX_CACHE_CONVERSATIONS)

    # PRIVATE FUNCTIONS
    def _test_connection(self):
        try:
            con = self.engine.connect()
            con.close()
        except:
            raise ConnectionError("Cannot connect to the provided connection string.")

    # PUBLIC FUNCTIONS
    def insert(self, inference_data: InferenceRequestDTO, user_name: str, last_message: str, action: str):
        new_session = self.session()
        new_operation = Operation(
            timestamp=datetime.now(),
            action=action,
            user_name=user_name,
            last_message=last_message,
            conversation=inference_data.messages.__repr__(),
            temperature=inference_data.inference_parameters.temperature,
            do_samples=inference_data.inference_parameters.do_sample,
            tokens_count=inference_data.inference_parameters.tokens_count,
            top_k_tokens=inference_data.inference_parameters.top_k_tokens,
            top_p_tokens=inference_data.inference_parameters.top_p_tokens
        )
        new_session.add(new_operation)
        new_session.commit()
        new_session.close()
        self.clear_cache()

    def clear_cache(self):
        new_session = self.session()
        operations_count = new_session.query(Operation).count()
        if operations_count > self.max_cache_conversations:
            rows_to_delete = operations_count - self.max_cache_conversations
            oldest_operations = new_session.query(Operation) \
                .order_by(Operation.timestamp.asc()) \
                .limit(rows_to_delete) \
                .all()
            for operation in oldest_operations:
                new_session.delete(operation)
            new_session.commit()
        new_session.close()

