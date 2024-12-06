import json

# Data to be converted to JSON
train_data = [
    ("What is the admission process for engineering colleges?", {'cats': {'admission_process': True, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),
    ("How do I apply for admission to a polytechnic college in Rajasthan?", {'cats': {'admission_process': True, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),
    ("Can you tell me the steps to apply for an engineering degree?", {'cats': {'admission_process': True, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),

    ("What are the eligibility criteria for engineering college admissions?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'eligibility_criteria': True}}),
    ("Do I need to have a specific score to get admitted to a college in Rajasthan?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'eligibility_criteria': True}}),

    ("Can you provide information on engineering colleges in Rajasthan?", {'cats': {'general_college_info': True, 'admission_process': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),
    ("How many engineering colleges are there in Rajasthan?", {'cats': {'general_college_info': True, 'admission_process': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),
    ("Which are the top colleges for engineering in Rajasthan?", {'cats': {'general_college_info': True, 'admission_process': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),

    ("What is the fee structure for engineering colleges in Rajasthan?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': True, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),
    ("How much do I need to pay for admission to a polytechnic college?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': True, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),
    ("Are there any additional fees for hostel facilities?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': True, 'scholarships': False, 'hostel_facilities': False, 'curriculum': False}}),

    ("What is the curriculum for engineering courses in Rajasthan?", {'cats': {'curriculum': True, 'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False}}),
    ("Can you tell me about the subjects in a civil engineering degree?", {'cats': {'curriculum': True, 'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False}}),

    ("Are there scholarships available for students in Rajasthan colleges?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': True, 'hostel_facilities': False, 'curriculum': False}}),
    ("What scholarships can I apply for in engineering colleges in Rajasthan?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': True, 'hostel_facilities': False, 'curriculum': False}}),

    ("What are the hostel facilities available in engineering colleges?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': True, 'curriculum': False}}),
    ("Can I stay in a college hostel, and what are the fees?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': True, 'curriculum': False}}),

    ("What are the job placement opportunities after graduating from engineering colleges in Rajasthan?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'placement_opportunities': True}}),
    ("Can you share information about recent placement statistics for engineering colleges?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'placement_opportunities': True}}),

    ("What was the cutoff for engineering colleges last year?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'cutoff': True}}),
    ("Can I get a list of colleges with their last year's admission statistics?", {'cats': {'admission_process': False, 'general_college_info': False, 'fee_structure': False, 'scholarships': False, 'hostel_facilities': False, 'cutoff': True}}),
]

# Save to JSON file
file_path = "train_data.json"
with open(file_path, 'w') as json_file:
    json.dump(train_data, json_file, indent=4)

file_path
