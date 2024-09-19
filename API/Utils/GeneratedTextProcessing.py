def process_generated_text(conversation_segments: str) -> list and str:
    dialogue = []
    gen_text = ""
    for segment in conversation_segments:
        new_segment = {}
        segment_split = segment.split("|>")
        if "system" in segment_split[0]:
            new_segment["role"] = 'system'
        if "user" in segment_split[0]:
            new_segment["role"] = 'user'
        if "assistant" in segment_split[0]:
            new_segment["role"] = 'system'
            gen_text = segment_split[1].strip()
        new_segment["content"] = segment_split[1].strip()
        dialogue.append(new_segment)
    return dialogue, gen_text