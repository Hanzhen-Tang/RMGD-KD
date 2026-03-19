"""统一导出教师模型和学生模型，便于单文件引用。"""

from models.student_gcn import SimpleGCNStudent
from models.teacher_gwnet import GWNetTeacher

__all__ = ["GWNetTeacher", "SimpleGCNStudent"]
