import json
from Google_Search_API_Wrapper.google_search_api import GoogleSearchAPI

data_file='path/to/2406_data'

def get_image_describe():
    prompt = f'''
    Please describe this image briefly in 100 words:
    '''
    return prompt

def get_video_describe():
    prompt = f'''
    Please describe this video:
    '''
    return prompt

def get_audio_describe():
    prompt = f'''
    Please describe this audio:
    '''
    return prompt


def get_search_summarize_prompt(s_result):
    prompt = f'''
    Please summarize the following search results briefly in 200 words:
    {s_result}
    '''
    return prompt

def WebSearchCheck(question, file_name, modality_type):
    from model.language_model import MedicalAssistant
    assistant = MedicalAssistant()

    if modality_type == 'text':
        qq = question
        print(qq)
        output = GoogleSearchAPI().response(method='text', max_results=3, query=qq)

    elif modality_type == 'image':
        p = get_image_describe()
        describe = assistant.generate_response(p, image_path=file_name)
        qq = describe + question
        print(qq)
        output = GoogleSearchAPI().response(method='text', max_results=3, query=qq)

    elif modality_type == 'video':
        from model.video_model import VideoLLaMAChatbot
        p = get_video_describe()
        bot = VideoLLaMAChatbot()
        describe = bot.chat(paths=file_name, text=p, modal_type='video')
        qq = describe + question
        print(qq)
        output = GoogleSearchAPI().response(method='text', max_results=3, query=qq)

    elif modality_type == 'audio':
        from model.audio_model import AudioChatbot
        p = get_audio_describe()
        bot = AudioChatbot()
        describe = bot.chat(audio=file_name, text=p)
        qq = describe + question
        print(qq)
        output = GoogleSearchAPI().response(method='text', max_results=3, query=qq)

    sp = get_search_summarize_prompt(output)
    search_r = assistant.generate_response(sp, max_new_tokens=256)
    return search_r
