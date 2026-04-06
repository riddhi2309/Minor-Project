from guard_llm import GuardLLM

class FullPipeline:
    def __init__(self, hf_token):
        self.guard_llm = GuardLLM(hf_token)

    def check(self, text):
        result = self.guard_llm.check(text)
        return result