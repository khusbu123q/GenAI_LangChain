from pydantic import BaseModel 

class Student (BaseModel):

    name :str

new_student ={"name":"Khusbu"}
student =Student(**new_student)
print(student)
student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()