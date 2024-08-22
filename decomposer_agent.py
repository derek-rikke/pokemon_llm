from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict

class DecomposerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.decomposition_chain = self._create_decomposition_chain()

    def _create_decomposition_chain(self):
        decomposition_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            Analyze the following Pokémon-related question and break it down into steps. For simple questions, use a single step. For complex questions, use multiple steps. Provide instructions for synthesizing the final answer.

            Question: {question}

            Steps:
            1. [First step or rephrased question]
            2. [Second step, if necessary]
            3. [Third step, if necessary]
            ...

            Synthesis Instructions: [Instructions for combining the results of the steps into a final answer]

            """
        )
        return LLMChain(llm=self.llm, prompt=decomposition_prompt)

    def decompose_question(self, question: str) -> Dict[str, List[str]]:
        result = self.decomposition_chain.run(question)
        lines = result.strip().split('\n')
        
        steps = []
        synthesis_instructions = ""
        
        for line in lines:
            if line.startswith(('Steps:', 'Step:')):
                continue
            elif line.startswith('Synthesis Instructions:'):
                synthesis_instructions = line.split(':', 1)[1].strip()
            elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                steps.append(line.split('.', 1)[1].strip())
        
        return {
            "steps": steps,
            "synthesis_instructions": synthesis_instructions
        }