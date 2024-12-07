import spacy
import json
import os

class IntentClassifier:
    def __init__(self, model_path='intent_model'):
        """
        Initialize the intent classifier
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = model_path
        self.nlp = None
    
    def train_model(self, train_data_path, force_retrain=False):
        """
        Train the model if not already trained
        
        Args:
            train_data_path (str): Path to training data JSON
            force_retrain (bool): Force retraining even if model exists
        """
        # Check if model already exists and we're not force retraining
        if os.path.exists(self.model_path) and not force_retrain:
            print("Model already exists. Loading existing model.")
            self.nlp = spacy.load(self.model_path)
            return
        
        # Load training data
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Create blank English model
        nlp = spacy.blank("en")
        text_classifier = nlp.add_pipe("textcat")
        
        # Collect all categories
        categories = set()
        for _, annotations in train_data:
            categories.update(annotations['cats'].keys())
        
        # Add categories to classifier
        for category in categories:
            text_classifier.add_label(category)
        
        # Prepare training examples
        train_examples = []
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = spacy.training.Example.from_dict(
                doc, 
                {"cats": annotations['cats']}
            )
            train_examples.append(example)
        
        # Initialize and train
        optimizer = nlp.begin_training()
        for _ in range(10):  # 10 epochs
            losses = {}
            nlp.update(train_examples, sgd=optimizer, losses=losses)
        
        # Save the trained model
        nlp.to_disk(self.model_path)
        print(f"Model trained and saved to {self.model_path}")
        
        self.nlp = nlp
    
    def predict(self, texts):
        """
        Predict intent categories for given texts
        
        Args:
            texts (list or str): Text or list of texts to classify
        
        Returns:
            list: Predictions for each text
        """
        if self.nlp is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = []
        for text in texts:
            doc = self.nlp(text)
            # Sort categories by confidence score
            sorted_cats = sorted(
                doc.cats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            predictions.append({
                'text': text,
                'categories': dict(sorted_cats)
            })
        
        return predictions

# Example usage
def main():
    # Initialize classifier
    classifier = IntentClassifier()
    
    # Train the model (will only train if model doesn't exist)
    classifier.train_model('train_data.json')
    
    # Test predictions
    test_texts = [
    # Admission-related
    "What is the admission process for undergraduate programs?",
    "Are there any entrance exams required for admission?",
    "How do I apply for admission online?",
    "What is the eligibility for applying to this course?",
    "Is there a direct admission option available?",
    "What documents are required for admission?",
    "Can I apply for multiple courses simultaneously?",
    "Is there a separate admission process for international students?",
    "Are there any specific age criteria for admission?",
    "What is the cutoff for admission this year?",
    "How is the merit list prepared?",
    "Can I apply without my final year exam results?",
    "Are admissions based on reservation or quota?",
    "What is the fee structure for admission?",
    "Is there an entrance exam for postgraduate programs?",
    "What is the duration of the admission process?",
    "When can I expect the result of the entrance exam?",
    "Is there any counseling process after the entrance exam?",
    "How can I withdraw my admission if needed?",
    "Can I apply for admission through the management quota?",

    # Scholarships and financial aid
    "Tell me about scholarship opportunities.",
    "How can I apply for a merit-based scholarship?",
    "What is the eligibility for need-based financial aid?",
    "Are there any scholarships for international students?",
    "How do I apply for a sports scholarship?",
    "Is there a scholarship for economically weaker students?",
    "How can I apply for government scholarships?",
    "What is the process to get a fee waiver?",
    "Are there any scholarships for students from specific regions?",
    "Are there any scholarships for students with disabilities?",
    "What is the process for renewing my scholarship every year?",
    "How do I check if I am eligible for a scholarship?",
    "Can I get a scholarship for my academic performance in the previous year?",
    "Is there a scholarship for research students?",
    "What is the maximum amount a student can get through scholarships?",
    "Are there any scholarships for first-generation learners?",
    "Can I apply for multiple scholarships at once?",
    "How long does it take to get the scholarship disbursed?",
    "Do I need to maintain a specific CGPA to keep my scholarship?",
    "Are there any external scholarships available for students?",

    # Academic programs
    "What are the courses offered in computer science?",
    "Can you suggest some elective courses for my semester?",
    "What is the syllabus for the first year of engineering?",
    "What are the career opportunities after completing a mechanical engineering degree?",
    "Is there an option for a double major?",
    "What are the specializations available in electrical engineering?",
    "What is the duration of a B.Tech program?",
    "Can I switch my major after the first year?",
    "What are the subjects covered in the MBA program?",
    "How can I change my course after enrollment?",
    "What is the difference between B.Tech and B.Sc. in Computer Science?",
    "How do I choose my electives for the next semester?",
    "Are there any interdisciplinary programs offered by the college?",
    "How many courses can I take in a semester?",
    "What is the credit system for academic courses?",
    "Is there an option for distance learning or online courses?",
    "What is the process for changing my course or specialization?",
    "How are the courses taught in this college, through lectures or projects?",
    "What is the average class size for each program?",
    "Are there any joint degree programs with other universities?",

    # Campus facilities
    "What are the hostel facilities?",
    "Is there a library on campus? What are its timings?",
    "Does the college have sports facilities?",
    "Are there any student clubs or organizations?",
    "What kind of food options are available in the hostel?",
    "Is there a Wi-Fi facility on campus?",
    "What are the transportation facilities for students?",
    "Does the campus have a gym or fitness center?",
    "Are there any common areas for group studies?",
    "Can I use the library during weekends?",
    "Are there separate hostels for boys and girls?",
    "What is the procedure to book a hostel room?",
    "Do I need to pay extra for using gym or sports facilities?",
    "How is the campus security for students?",
    "Is there a medical facility on campus?",
    "What is the campus dress code?",
    "Is there a parking facility on campus for students?",
    "What kind of extracurricular activities are organized on campus?",
    "Is there a bank or ATM on campus?",
    "Are there any international student services or accommodations?",

    # Career and placement
    "What are the placement statistics for this college?",
    "Which companies visit the campus for recruitment?",
    "Is there an internship program in the curriculum?",
    "How does the college help with career counseling?",
    "Are there on-campus recruitment drives?",
    "What is the average salary package offered during placements?",
    "How can I prepare for campus recruitment?",
    "Does the college offer job placement assistance after graduation?",
    "Is there a career fair or job fair held on campus?",
    "What is the alumni network like, and how does it help with placements?",
    "Are there any workshops or training sessions for interview preparation?",
    "Can I apply for internships abroad through the college?",
    "What is the procedure to apply for off-campus placements?",
    "How does the college assist students in building a resume?",
    "Is there any placement guarantee offered by the college?",
    "Are there any specialized courses for career skill development?",
    "Do I need to register for placement services separately?",
    "What industries does the college have the strongest placement ties with?",
    "How long does the placement process take after graduation?",
    "Is there any career support available for entrepreneurs or startups?",

    # Exams and results
    "When will the semester exams be held?",
    "How can I check my exam results?",
    "What is the procedure for re-evaluation of marks?",
    "Is there a provision for supplementary exams?",
    "Can I apply for an exam postponement?",
    "How can I get a duplicate exam mark sheet?",
    "Are there any online resources for exam preparation?",
    "What is the pass percentage required for the semester exam?",
    "How are marks calculated in this college?",
    "Can I request to change my exam center?",
    "How is the final grade calculated in the semester?",
    "When are the mid-term exams conducted?",
    "Are there any make-up exams available?",
    "What happens if I miss an exam due to illness?",
    "Can I take an exam in a different language?",
    "What is the process to get exam hall tickets?",
    "Are there any specific exam rules I need to follow?",
    "Can I carry a calculator or other tools during exams?",
    "How can I challenge my exam results if I think there is an error?",
    "Is there a specific dress code for exams?",

    # Fees and payments
    "What is the fee structure for undergraduate courses?",
    "Can I pay the fees in installments?",
    "Are there any additional fees for lab facilities?",
    "What is the refund policy if I withdraw admission?",
    "Is there an option to pay fees online?",
    "How can I get a fee receipt?",
    "Are there any late payment penalties for fees?",
    "Can I pay my fees using a credit card?",
    "Is there any concession on fees for students with financial needs?",
    "What happens if I miss the fee payment deadline?",
    "Are there any scholarship programs to reduce tuition fees?",
    "Can I get a loan for paying my fees?",
    "What is the last date to pay the semester fees?",
    "Are there any fees for extra-curricular activities?",
    "Can I get an exemption from paying the hostel fees?",
    "How can I check my pending fee payments?",
    "Is there a payment gateway for international students?",
    "Are there any discounts on fees for early payment?",
    "Is there any separate fee structure for foreign students?",
    "Can I pay my fees in foreign currency?",

    # Campus life
    "What is the average student life like here?",
    "Are there any cultural festivals or events organized?",
    "How safe is the campus for students?",
    "What is the best way to make friends in college?",
    "How can I get involved in extracurricular activities?",
    "Are there any opportunities for volunteer work?",
    "What are the common hangout spots on campus?",
    "Are there any annual fests or cultural events?",
    "Can I participate in both academic and non-academic clubs?",
    "How are the social interactions between students of different programs?",
    "What are the most popular student events here?",
    "Does the college organize any sports tournaments?",
    "Are there any student exchange programs?",
    "How can I participate in research groups or projects?",
    "What is the overall student diversity in this college?",
    "Are there any alumni meetups or gatherings?",
    "How does the college promote mental health and well-being?",
    "Are there any support systems for freshmen?",
    "Are there any international student clubs on campus?",
    "What is the student-to-faculty ratio?",

    # Technical queries
    "How can I access the college's Wi-Fi network?",
    "Is there an online portal for submitting assignments?",
    "How can I log in to my student account?",
    "What should I do if I forget my student portal password?",
    "How can I update my contact details in the student database?",
    "How do I access the online library resources?",
    "What are the system requirements for using the college portal?",
    "Is there a mobile app for the college?",
    "How can I request for a computer lab booking?",
    "Can I access my class materials online?",
    "How do I download my course syllabus from the portal?",
    "What to do if the college portal is not working?",
    "How do I submit my project or thesis online?",
    "How can I get access to the course material in my email?",
    "Can I use personal software in the campus labs?",
    "How do I register for online classes?",
    "What should I do if I face technical issues during an online exam?",
    "Are there any online platforms for peer-to-peer learning?",
    "Can I collaborate with classmates on projects through the online system?",
    "How can I get IT support if I face technical issues?"
]
    
    predictions = classifier.predict(test_texts)
    
    # Print predictions
    for pred in predictions:
        print(f"\nText: {pred['text']}")
        print("Intent:")
        for cat, score in pred['categories'].items():
            if score > 0.5:  # Only show categories with >50% confidence
                print(f"{cat}: {score:.2f}")

if __name__ == "__main__":
    main()