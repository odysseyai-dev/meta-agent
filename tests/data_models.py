from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict, Annotated

class ResumeModel(BaseModel):
    raw_text: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    work_history: List[Dict[str, Any]] = Field(default_factory=list)

class SuggestionsModel(BaseModel):
    suggestions_list: List[str] = Field(default_factory=list)
    is_approved: Optional[bool] = None

class JobListingModel(BaseModel):
    id: str
    title: str
    description: str

class JobAnalysisModel(BaseModel):
    job_id: str
    fit_score: int
    mismatch_analysis: str

class ApplicationStatusModel(BaseModel):
    job_id: str
    status: str

class UserPreferences(BaseModel):
    target_roles: List[str] = Field(default_factory=list)
    min_salary: Optional[int] = None