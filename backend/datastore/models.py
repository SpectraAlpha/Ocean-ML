"""
Datastore module for IoT image and data stream ingestion
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from sqlalchemy import Column, String, DateTime, JSON, Float, Integer, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field


Base = declarative_base()


class DataType(str, Enum):
    """Type of data ingested"""
    IMAGE = "image"
    SENSOR = "sensor"
    VIDEO = "video"
    METADATA = "metadata"


class DataSource(str, Enum):
    """Source of IoT data"""
    IOT_CAMERA = "iot_camera"
    SATELLITE = "satellite"
    DRONE = "drone"
    BUOY_SENSOR = "buoy_sensor"
    VESSEL_SENSOR = "vessel_sensor"


class DataEntry(Base):
    """Database model for data entries"""
    __tablename__ = "data_entries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    data_type = Column(String, nullable=False)
    source = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    location_lat = Column(Float, nullable=True)
    location_lon = Column(Float, nullable=True)
    file_path = Column(String, nullable=True)
    s3_key = Column(String, nullable=True)
    metadata = Column(JSON, default={})
    tags = Column(JSON, default=[])
    llm_tags = Column(JSON, default=[])
    llm_description = Column(Text, nullable=True)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataEntryCreate(BaseModel):
    """Schema for creating data entries"""
    data_type: DataType
    source: DataSource
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    file_path: Optional[str] = None
    s3_key: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class DataEntryResponse(BaseModel):
    """Schema for data entry responses"""
    id: str
    data_type: str
    source: str
    timestamp: datetime
    location_lat: Optional[float]
    location_lon: Optional[float]
    file_path: Optional[str]
    s3_key: Optional[str]
    metadata: Dict[str, Any]
    tags: List[str]
    llm_tags: List[str]
    llm_description: Optional[str]
    processed: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DataStore:
    """Main datastore class for managing IoT data"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_entry(self, entry_data: DataEntryCreate) -> DataEntry:
        """Create a new data entry"""
        entry = DataEntry(
            data_type=entry_data.data_type.value,
            source=entry_data.source.value,
            timestamp=entry_data.timestamp,
            location_lat=entry_data.location_lat,
            location_lon=entry_data.location_lon,
            file_path=entry_data.file_path,
            s3_key=entry_data.s3_key,
            metadata=entry_data.metadata,
            tags=entry_data.tags
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry
    
    def get_entry(self, entry_id: str) -> Optional[DataEntry]:
        """Retrieve a data entry by ID"""
        return self.db.query(DataEntry).filter(DataEntry.id == entry_id).first()
    
    def list_entries(
        self,
        skip: int = 0,
        limit: int = 100,
        data_type: Optional[DataType] = None,
        source: Optional[DataSource] = None,
        processed: Optional[bool] = None
    ) -> List[DataEntry]:
        """List data entries with filters"""
        query = self.db.query(DataEntry)
        
        if data_type:
            query = query.filter(DataEntry.data_type == data_type.value)
        if source:
            query = query.filter(DataEntry.source == source.value)
        if processed is not None:
            query = query.filter(DataEntry.processed == processed)
        
        return query.offset(skip).limit(limit).all()
    
    def update_tags(self, entry_id: str, tags: List[str]) -> Optional[DataEntry]:
        """Update tags for a data entry"""
        entry = self.get_entry(entry_id)
        if entry:
            entry.tags = tags
            self.db.commit()
            self.db.refresh(entry)
        return entry
    
    def add_llm_enrichment(
        self,
        entry_id: str,
        llm_tags: List[str],
        llm_description: Optional[str] = None
    ) -> Optional[DataEntry]:
        """Add LLM-generated tags and description"""
        entry = self.get_entry(entry_id)
        if entry:
            entry.llm_tags = llm_tags
            entry.llm_description = llm_description
            self.db.commit()
            self.db.refresh(entry)
        return entry
    
    def mark_processed(self, entry_id: str) -> Optional[DataEntry]:
        """Mark entry as processed"""
        entry = self.get_entry(entry_id)
        if entry:
            entry.processed = True
            self.db.commit()
            self.db.refresh(entry)
        return entry
