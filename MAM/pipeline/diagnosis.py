def get_diagnosis_prompt(ques,record):
    prompt = f'''
    Input: Based on the provided image/video/audio (if applicable) and the meeting record, please provide answer to the following question.

    Question: {ques}.

    Meeting record: {record}
    '''

    return prompt


def final_diagnosis(question, file_name, modality_type, type_name, meeting_record):
    from model.language_model import MedicalAssistant
    assistant = MedicalAssistant()
    query = get_diagnosis_prompt(question, meeting_record)
    if modality_type == 'image':
        output = assistant.generate_response(query, image_path=file_name)
    elif modality_type == 'audio':
        from model.audio_model import AudioChatbot
        bot = AudioChatbot()
        output = bot.chat(audio=file_name, text=query)
    elif modality_type == 'video':
        from model.video_model import VideoLLaMAChatbot
        bot = VideoLLaMAChatbot()
        output = bot.chat(paths=file_name, text=query, modal_type='video')
    elif modality_type == 'text':
        output = assistant.generate_response(query)
    print(f"\n--- final diagnosis ---")
    print(output)
    return output
