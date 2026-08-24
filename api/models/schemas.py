from sqlalchemy import Column, Integer, String, Text, DateTime
from api.database import Base
import datetime

class SummaryRecord(Base):
    __tablename__ = "summary_records"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    original_text = Column(Text)
    summary = Column(Text)
    style = Column(String)
    length = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
