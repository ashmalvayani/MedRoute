def generate_role(disease_type, modality_type, question, file_name):
    from model.language_model import MedicalAssistant
    system_prompt = f"Given a disease type, generate a system prompt that assigns tasks to relevant medical roles, including **Specialist Doctor**, **Radiologic Technologist**, etc, from the perspective of a General Practitioner."
    user_prompt = f'''Input: The modality type is {modality_type}, the disease type is {disease_type}, and the patient question is {question}.
    Output:
    A system prompt that:
        Identifies the relevant Specialist Doctor(s), Radiologic Technologist(s), and other Specialist(s) for the given disease type.
        Assigns tasks to each identified role, specifying the necessary actions, tests, or examinations required for diagnosis and treatment.

    IMPORTANT: Output ONLY the roles in the exact format below. Do NOT include any preamble, headers, "System Prompt:", markdown headings (###), or explanations before the roles. Start directly with the first **Specialist Doctor** line. Each role header must end with a colon immediately followed by a newline. Do NOT put any spaces or whitespace between the colon and the newline.

    Output example:
        **Specialist Doctor** (Pulmonologist):
        - Assess Patient's Health: Evaluate patient's function and overall health
        - Use {modality_type} Studies: Utilize expertise in the {disease_type} domains to diagnose diseases
        - Analyze Patient History and Symptoms: Determine the cause and severity of diseases by analyzing patient's medical history and symptoms
    '''
    
    assistant = MedicalAssistant()
    response = assistant.generate_response_role_gen(system_prompt + "\n" + user_prompt)
    #print(response)
    return response
