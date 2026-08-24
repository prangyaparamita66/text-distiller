from sqlalchemy.orm import Session
from api.models import schemas

def get_summaries(db: Session, skip: int = 0, limit: int = 100):
    return db.query(schemas.SummaryRecord).order_by(schemas.SummaryRecord.created_at.desc()).offset(skip).limit(limit).all()

def create_summary(db: Session, title: str, original_text: str, summary: str, style: str, length: str):
    db_summary = schemas.SummaryRecord(
        title=title,
        original_text=original_text,
        summary=summary,
        style=style,
        length=length
    )
    db.add(db_summary)
    db.commit()
    db.refresh(db_summary)
    return db_summary

def delete_summary(db: Session, summary_id: int):
    db_summary = db.query(schemas.SummaryRecord).filter(schemas.SummaryRecord.id == summary_id).first()
    if db_summary:
        db.delete(db_summary)
        db.commit()
        return True
    return False
