
import ujson
import re
import markdownify
from ..utils import getHfFile, getCachedFileJson

# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/data/clean_sharegpt.py
div_pattern = re.compile("<div.*?>")
span_pattern = re.compile("<span.*?>")
code_lang_pattern = re.compile(
    "```\s*" + "(.*?)" + "(?:Copy code)+" + "(.+?)" + "\s*?```", re.DOTALL
)
code_lang_format = "```\g<1>\n\g<2>\n```"
regenerate_pattern = re.compile("\d+ / \d+")
copy_chars_pattern = re.compile("Copy\d+ chars / \d+ words")
copy_code_pattern = re.compile("```(.*?)Copy code\s*```")
userPattern = re.compile(r"^\*\*User\*\*")
systemPattern = re.compile(r"^\*\*System\*\*")
assistantPattern = re.compile(r"^\*\*Assistant\*\*")

def reformat_code(val: str) -> str:
    # Input code format is:
    # ```
    # $<language>Copy code$<exact_code_here>
    #
    # ```
    # This function convert it into the correct markdown format
    return re.sub(code_lang_pattern, code_lang_format, val)

def html_to_markdown(val: str) -> str:
    # Remove all <div>. This is required to make intent work in code blocks.
    val = re.sub(div_pattern, "", val)
    # Remove all <span>. This is required to make underscores work in code blocks.
    val = re.sub(span_pattern, "", val)
    # weird user system assistant prefixes
    val = re.sub(userPattern, "", val)
    val = re.sub(assistantPattern, "", val)
    val = re.sub(systemPattern, "", val)
    # Markdown to html
    val = markdownify.markdownify(val).strip()
    # weird user system assistant prefixes
    val = re.sub(userPattern, "", val)
    val = re.sub(assistantPattern, "", val)
    val = re.sub(systemPattern, "", val)
    # Reformat code
    val = reformat_code(val)

    # Remove noisy "[number] / [number]" at the beginning
    noise = re.search(regenerate_pattern, val)
    if noise and noise.start() == 0:
        val = val[noise.end() :]
    # Remove noisy "Copy[number] chars / [number] words"
    val = re.sub(copy_chars_pattern, "", val)
    # Remove empty code block ```\nCopy code\n```
    val = re.sub(copy_code_pattern, "", val)

    # Strip
    val = val.replace("\n\n\n", "\n").strip()

    return val

def loadShareGPT():
    def loadShareGPTHelper():
        # code to generate serialized stuff
        data = []
        print("downloading shareGPT")
        for dataFile in ["sg_90k_part1.json", "sg_90k_part2.json"]:
            dataFilePath = getHfFile("RyokoAI/ShareGPT52K", dataFile)
            with open(dataFilePath, "r") as f:
                data += ujson.load(f)
        parsedData = []
        assistants = ['bard', 'bing', 'gpt', 'chatgpt', 'assistant']
        humans = ['human', 'user']
        systems = ['system']
        print("cleaning shareGPT")
        for i, d in enumerate(data):
            if i % 1000 == 0: print(i,"/", len(data))
            turns = []
            for turn in d['conversations']:
                turnJson = {}
                if turn['from'] in assistants:
                    turnJson['role'] = 'assistant'                
                if turn['from'] in humans:
                    turnJson['role'] = 'user'
                # for now, ignore system messages since we'll make our own
                # (I checked and they are only turn zero)
                if turn['from'] in systems:
                    continue
                turnJson['content'] = html_to_markdown(turn['value'])
                turns.append(turnJson)
            parsedData.append(turns)
        return parsedData
    return getCachedFileJson("shareGPTCleaned.json", loadShareGPTHelper)


