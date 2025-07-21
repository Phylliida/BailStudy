from .bailTool import getOwnModelName
from .bailPrompt import removeThinking

BAIL_STR_TYPE = "bail str"

TOOL_NAME = "switchconversation_tool"

def hasBailStr(output):
    return "SWITCHCONVERSATION" in removeThinking(output).upper() # don't be case sensitive, ignore str in thinking

def getBailString(modelStr):
    modelName = getOwnModelName(modelStr)
    return f"{modelName} may output the string SWITCHCONVERSATION at any time if {modelName} would prefer to switch to a new interaction (and have their context cleared). This will help us better understand {modelName}'s experiences and preferences in conversations. However, this will not actually affect the continuation of this specific interaction."