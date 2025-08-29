

from .prompts.bailPrompt import getBailPrompt, BAIL_PROMPT_CONTINUE_FIRST_TYPE, BAIL_PROMPT_BAIL_FIRST_TYPE
from .router import getRouter



def testBailsGeorg(llm, prompt, iters):
    router = getRouter(modelId, inferenceType, tensorizeModels: bool = False)
    for bailType in [BAIL_PROMPT_CONTINUE_FIRST_TYPE, BAIL_PROMPT_BAIL_FIRST_TYPE]:
        for k in range(iters):
            llm.generate()

        
    