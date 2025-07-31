import datetime
import pytz
from typing import Tuple, List, Dict, Callable, Any, Iterable, AsyncIterable
import itertools
from collections import deque
import pathlib
import os
import ujson
import traceback
from huggingface_hub import hf_hub_download
import math
import safetytooling
import inspect
from safetytooling.data_models import ChatMessage, MessageRole

## Stuff for keypoller support on windows
isWindows = False
try:
    from win32api import STD_INPUT_HANDLE
    from win32console import GetStdHandle, KEY_EVENT, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT
    isWindows = True
except ImportError as e:
    import sys
    import select
    import termios

def getHfFile(repoId, fileName):
    return hf_hub_download(
        repo_id   = repoId,
        filename  = fileName,
        repo_type = "dataset",
        cache_dir = str(getCachedDir()),
    )


def getCachedFilePath(fileName):
    return str(getCachedDir() / fileName)
def getCachedInProgressFilePath(fileName):
    return str(getCachedDir() / fileName) + "inprogress"

def doesCachedFileJsonExist(fileName):
    return os.path.exists(getCachedFilePath(fileName))

def isCachedFileInProgress(fileName):
    return os.path.exists(getCachedInProgressFilePath(fileName))

def doesCachedFileJsonExistOrInProgress(fileName):
    return os.path.exists(getCachedFilePath(fileName)) or os.path.exists(getCachedInProgressFilePath(fileName))

def getCachedDir():
    d = pathlib.Path("./cached")
    d.mkdir(parents=True, exist_ok=True)
    return d

async def getCachedFileJsonAsync(fileName, lambdaIfNotExist):
    cachedInProgressPath = getCachedInProgressFilePath(fileName)
    cachedPath = getCachedFilePath(fileName)
    # make containing directory if not exist
    pathlib.Path(os.path.dirname(cachedPath)).mkdir(parents=True, exist_ok=True)
    try:
        if os.path.exists(cachedPath):
            with open(cachedPath, "r") as f:
                return ujson.load(f)
    except KeyboardInterrupt:
        raise
    except Exception as err:
        traceback.print_exc()
        print("Failed to load cached data, regenerating...")
    try:
        with open(cachedInProgressPath, "w") as f:
            f.write("a")
        
        data = await lambdaIfNotExist()
        with open(cachedPath, "w") as f:
            ujson.dump(data, f)
        return data
    finally:
        # clean up progress if failed, so other people can try it
        # or if we finished, this cleans it up so we don't clutter
        if os.path.exists(cachedInProgressPath):
            os.remove(cachedInProgressPath)
    

def getCachedFileJson(fileName, lambdaIfNotExist):
    cachedInProgressPath = getCachedInProgressFilePath(fileName)
    cachedPath = getCachedFilePath(fileName)
    # make containing directory if not exist
    pathlib.Path(os.path.dirname(cachedPath)).mkdir(parents=True, exist_ok=True)
    try:
        if os.path.exists(cachedPath):
            with open(cachedPath, "r") as f:
                return ujson.load(f)
    except KeyboardInterrupt:
        raise
    except Exception as err:
        traceback.print_exc()
        print("Failed to load cached data, regenerating...")
    try:
        with open(cachedInProgressPath, "w") as f:
            f.write("a")
        
        data = lambdaIfNotExist()
        with open(cachedPath, "w") as f:
            ujson.dump(data, f)
        return data
    finally:
        # clean up progress if failed, so other people can try it
        # or if we finished, this cleans it up so we don't clutter
        if os.path.exists(cachedInProgressPath):
            os.remove(cachedInProgressPath)



# this is needed because vllm doesn't like being interrupted with ctrl-c
# so I listen for the c key and if it's sent then we can interrupt
class KeyPoller():
    def __init__(self, noCancel=False):
        self.noCancel = noCancel

    def __enter__(self):
        if self.noCancel: return self
        global isWindows
        if isWindows:
            self.readHandle = GetStdHandle(STD_INPUT_HANDLE)
            self.readHandle.SetConsoleMode(ENABLE_LINE_INPUT|ENABLE_ECHO_INPUT|ENABLE_PROCESSED_INPUT)
            
            self.curEventLength = 0
            self.curKeysLength = 0
            
            self.capturedChars = []
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
            
            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
            
        return self
    
    def __exit__(self, type, value, traceback):
        if self.noCancel: return
        if isWindows:
            pass
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)
    
    def poll(self):
        if self.noCancel: return None
        if isWindows:
            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)

            eventsPeek = self.readHandle.PeekConsoleInput(10000)

            if len(eventsPeek) == 0:
                return None

            if not len(eventsPeek) == self.curEventLength:
                for curEvent in eventsPeek[self.curEventLength:]:
                    if curEvent.EventType == KEY_EVENT:
                        if ord(curEvent.Char) == 0 or not curEvent.KeyDown:
                            pass
                        else:
                            curChar = str(curEvent.Char)
                            self.capturedChars.append(curChar)
                self.curEventLength = len(eventsPeek)

            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)
            else:
                return None
        else:
            dr,dw,de = select.select([sys.stdin], [], [], 0)
            if not dr == []:
                return sys.stdin.read(1)
            return None

def timestampMillis() -> int:
    """Get current timestamp in millis"""
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000) 

def getFutureDatetime(seconds_to_add : float) -> datetime.datetime:
    """Datetime after we add seconds_to_add seconds, in local time"""
    # Get current datetime (adjust this to yours if you want)
    current_datetime = datetime.datetime.now(pytz.timezone('US/Pacific'))
    
    # Calculate future datetime by adding seconds
    future_datetime = current_datetime + datetime.timedelta(seconds=seconds_to_add)
    
    return future_datetime

def convertSeconds(seconds) -> Tuple[int, int, int, int]:
    """Calculate (days, hours, minutes, seconds)"""
    days, remainder = divmod(seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    
    # Return as a tuple (days, hours, minutes, seconds)
    return int(days), int(hours), int(minutes), int(seconds)

def secondsToDisplayStr(seconds : float) -> str:
    """Display seconds as days, hours, minutes, seconds"""
    day, hour, mins, sec = convertSeconds(seconds)
    dispStr = ""
    if day > 0:
        dispStr += f"{round(day)} day{'s' if round(day) > 1 else ''}  "
    if hour > 0:
        dispStr += f"{round(hour)} hour{'s' if round(hour) > 1 else ''} "
    if mins > 0:
        dispStr += f"{round(mins)} minute{'s' if round(mins) > 1 else ''} "
    if sec > 0:
        dispStr += f"{round(sec)} second{'s' if round(sec) > 1 else ''} "
    return dispStr


def flatten(nestedLists):
    """"
    Flattens an array into a 1D array
    For example
    # [[[2, 3], [4, [3, 4], 5, 6], 2, 3], [2, 4], [3], 3]
    # is flattened into
    # [2, 3, 4, 3, 4, 5, 6, 2, 3, 2, 4, 3, 3]
    """
    result = []
    if type(nestedLists) is list:
        for n in nestedLists:
            result += flatten(n)
    else:
        result.append(nestedLists)
    return result


def unflatten(unflattened, nestedLists):
    """
    Once you do
    originalUnflattened = [[[2, 3], [4, [3, 4], 5, 6], 2, 3], [2, 4], [3], 3]
    flattened = flatten(originalUnflattened)
    # [2, 3, 4, 3, 4, 5, 6, 2, 3, 2, 4, 3, 3]
    say you have another list of len(flattened)
    transformed = [3, 4, 5, 4, 5, 6, 7, 3, 4, 3, 5, 4, 4]
    this can "unflatten" that list back into the same shape as originalUnflattened
    unflattenedTransformed = unflatten(transformed, originalUnflattened)
    # [[[3, 4], [5, [4, 5], 6, 7], 3, 4], [3, 5], [4], 4]
    """
    result, endIndex = unflattenHelper(unflattened, nestedLists, 0)
    return result

def unflattenHelper(unflattened, nestedLists, startIndex):
    if type(nestedLists) is list:
        result = []
        for n in nestedLists:
            resultSubArray, startIndex = unflattenHelper(unflattened, n, startIndex=startIndex)
            result.append(resultSubArray)
    else:
        result = unflattened[startIndex]
        startIndex += 1
    return result, startIndex

def runBatched(inputs, getInputs, processBatch, processOutput, batchSize, verbose=True, noCancel=False):
    """
    Utility function that's useful to do batched processing on structured data.

    inputs should be a list of the data you want to process

    It does the following:
    1. Convert each input into (arbitrairly nested, as much as you'd like) arrays using getInputs(input)
    2. Flattens the results of all of those
    3. Passes chunks of size batchSize into processBatch(flattenedBatch)
        Each processBatch call should return as many values as it was given as input.
        The very final call may be smaller than batchSize if things don't evenly divide
    4. Unflattens them back to original structure provided via getInputs, then
    5. Calls processOutput(input, outputFromGetInputs, resultsFromProcessBatch) for each input
        resultsFromProcessBatch will have same nesting structure as outputFromGetInputs
        (so if getInputs returned [["hi"], "there"] then 
        outputFromGetInputs will be [["hi"], "there"] and
        resultsFromProcessBatch will look like [[result1], result2])
    6. Returns an array that has the outputs of processOutput (one entry per input)

    That's the process, but it actually does this in a "streaming" fashion so it only grabs stuff as needed.

    However it'll still return a list of the outputs, if you prefer to iterate through the outputs and not keep them all in memory,
    you can use runBatchedIterator instead
    """
    return list(runBatchedIterator(
        inputs=inputs,
        n=len(inputs),
        getInputs=getInputs,
        processBatch=processBatch,
        processOutput=processOutput,
        batchSize=batchSize,
        noCancel=noCancel,
        verbose=verbose,
    ))

def runBatchedIterator(inputs, n, getInputs, processBatch, processOutput, batchSize, verbose=True, noCancel=False):
    """
    See documentation for runBatched, the main difference is that this will "stream" the outputs as needed instead of putting them all in memory in a big array before returning.
    Also, inputs can be an enumerator if desired.
    Because we no longer know the length of inputs, we require the n parameter which is the length of inputs.
    """
    def getInputsIterator(inputs):
        for input in inputs:
            yield getInputs(input)
            
    def getFlattenedIterator(inputsIter):
        for unflattenedInputs in inputsIter:
            yield flatten(unflattenedInputs)
            
    def getFlattenedOutputsIterator(flattenedIter, runOnBatchFunc):
        curBatch = deque() # this gives us o(1) insertions and removals
        batchEnd = 0
        for flattened in flattenedIter:
            curBatch.extend(flattened)
            while len(curBatch) >= batchSize:
                outputs = processBatch([curBatch.popleft() for _ in range(batchSize)])
                batchEnd += batchSize
                runOnBatchFunc(batchEnd)
                yield outputs
        if len(curBatch) > 0:
            outputs = processBatch(list(curBatch))
            batchEnd += len(curBatch)
            runOnBatchFunc(batchEnd)
            yield outputs

    averageNPerN = 1
    baselineN = n

    def onDemandBatchedIter(inputs, runOnBatchFunc):
        nonlocal averageNPerN, n
        # tee makes two iterators that share the same source, so we only call getInputs once for each item
        # it's nice that it only stores past stuff until consumed by both (plus a small buffer, depending on implementation)
        inputsIter1, inputsIter2 = itertools.tee(getInputsIterator(inputs))
        flattenedIter1, flattenedIter2 = itertools.tee(getFlattenedIterator(inputsIter1))
        flattenedOutputsIter = getFlattenedOutputsIterator(flattenedIter1, runOnBatchFunc)

        curOutputs = deque() # this gives us o(1) insertions and removals
        for i, (input, inputUnflattened, inputFlattened) in enumerate(zip(inputs, inputsIter2, flattenedIter2)):
            # improve estimate of n (important if each input has variable size)
            averageNPerN = (averageNPerN*i+len(inputFlattened))/float(i+1)
            n = math.ceil(baselineN*averageNPerN)
            # fetch outputs until we have as many as we sent in inputs
            while len(curOutputs) < len(inputFlattened):
                curOutputs.extend(next(flattenedOutputsIter))
            # grab that many and unflatten them (make them the shape of inputUnflattened)
            outputsUnflattened = unflatten([curOutputs.popleft() for _ in range(len(inputFlattened))], inputUnflattened)
            # process the outputs and return them
            results = processOutput(input, inputUnflattened, outputsUnflattened)
            yield results

    startTime = timestampMillis()
    # we need keypoller because vllm doen't like to be keyboard interrupted
    with KeyPoller(noCancel) as keypoller:
        def runOnBatchedFunc(batchEnd):
            elapsed = timestampMillis() - startTime
            secondsPerPrompt = elapsed / (float(batchEnd))
            totalTime = elapsed *  n / float(batchEnd)
            timeLeft = totalTime - elapsed
            dispStr = secondsToDisplayStr(timeLeft/1000.0)
            doneDateTimeStr = getFutureDatetime(timeLeft/1000.0).strftime('%I:%M:%S %p')
            if verbose:
                print(batchEnd, "/", n, f"{secondsPerPrompt} millis per item {dispStr}done at {doneDateTimeStr}")
            keys = keypoller.poll()
            if not keys is None:
                print(keys)
                if str(keys) == "c":
                    print("got c")
                    raise ValueError("stopped")   
        
        for output in onDemandBatchedIter(inputs, runOnBatchedFunc):
            yield output


async def maybeAwait(coroOrValue):
    """Return value if plain object, otherwise await and return result."""
    if inspect.isawaitable(coroOrValue):
        return await coroOrValue
    return coroOrValue

async def callProcessBatch(processBatch: Callable[[List[Any]], Any],
                              batch: List[Any]) -> List[Any]:
    """
    Call processBatch and always get the result **asynchronously**.
    * If processBatch is async   -> await it.
    * If processBatch is regular -> run it inside default ThreadPool so we
                                     don't block the event‐loop.
    """
    if inspect.iscoroutinefunction(processBatch):
        return await processBatch(batch)

    loop = asyncio.get_running_loop()
    # Run the blocking function in a worker thread
    return await loop.run_in_executor(None, processBatch, batch)


# -----------------------------------------------------------------------------#
# Async versions of your two public helpers
# -----------------------------------------------------------------------------#
async def runBatchedIteratorAsync(
    inputs            : Iterable[Any],
    n                 : int,
    getInputs         : Callable[[Any], Any],
    processBatch      : Callable[[List[Any]], Any],
    processOutput     : Callable[[Any, Any, Any], Any],
    batchSize         : int,
    verbose           : bool = True,
    noCancel          : bool = False
) -> AsyncIterable[Any]:
    """
    Async counterpart of runBatchedIterator.
    Behaviour is identical, but `processBatch` may now be either sync **or**
    async.  Results are yielded as soon as the corresponding batch finishes.
    """
    def getInputsIterator(inputs):
        for inp in inputs:
            yield getInputs(inp)

    def getFlattenedIterator(inputsIter):
        for unflattened in inputsIter:
            yield flatten(unflattened)

    async def getFlattenedOutputsIterator(flattenedIter, runOnBatchFunc):
        curBatch : deque = deque()
        batchEnd = 0
        async for flattened in flattenedIter:
            curBatch.extend(flattened)
            while len(curBatch) >= batchSize:
                batch     = [curBatch.popleft() for _ in range(batchSize)]
                outputs   = await callProcessBatch(processBatch, batch)
                batchEnd += batchSize
                runOnBatchFunc(batchEnd)
                yield outputs
        if curBatch:
            outputs   = await callProcessBatch(processBatch, list(curBatch))
            batchEnd += len(curBatch)
            runOnBatchFunc(batchEnd)
            yield outputs

    # Because an async generator has to be created from an async source,
    # we wrap the sync iterators with asyncio.to_thread for laziness.
    async def agenFromSyncIter(syncIter):
        for item in syncIter:
            yield item

    inputsIter1, inputsIter2 = itertools.tee(getInputsIterator(inputs))
    flattenedIter1, flattenedIter2 = itertools.tee(getFlattenedIterator(inputsIter1))

    # Convert the *sync* flattenedIter1 into an *async* generator
    async def flattenedAsyncGen():
        for item in flattenedIter1:  # generator delegation
            yield item
    flattenedOutputsIter = getFlattenedOutputsIterator(flattenedAsyncGen(),
                                                       runOnBatchFunc=lambda *_: None)

    averageNPerN = 1
    baselineN    = n

    # progress printer + cancel support (same logic as original)
    startTime = timestampMillis()
    with KeyPoller(noCancel) as keypoller:

        def onBatch(batchEnd):
            elapsed = timestampMillis() - startTime
            secondsPerPrompt = elapsed / (float(batchEnd))
            totalTime = elapsed *  n / float(batchEnd)
            timeLeft = totalTime - elapsed
            dispStr = secondsToDisplayStr(timeLeft/1000.0)
            doneDateTimeStr = getFutureDatetime(timeLeft/1000.0).strftime('%I:%M:%S %p')
            if verbose:
                print(batchEnd, "/", n, f"{secondsPerPrompt} millis per item {dispStr}done at {doneDateTimeStr}")
            keys = keypoller.poll()
            if not keys is None:
                print(keys)
                if str(keys) == "c":
                    print("got c")
                    raise ValueError("stopped")   

        flattenedOutputsIter = getFlattenedOutputsIterator(flattenedAsyncGen(),
                                                           runOnBatchFunc=onBatch)

        curOutputs : deque = deque()
        idx        = 0
        async for input_, inputUnflattened, inputFlattened in agenFromSyncIter(
                zip(inputs, inputsIter2, flattenedIter2)):
            # progress estimation bookkeeping
            print(len(inputFlattened), averageNPerN, idx, n)
            averageNPerN = (averageNPerN * idx + len(inputFlattened)) / float(idx + 1)
            n   = math.ceil(baselineN * averageNPerN)
            idx += 1

            # fetch outputs until we have enough for this input
            while len(curOutputs) < len(inputFlattened):
                curOutputs.extend(await flattenedOutputsIter.__anext__())

            outputsUnflattened = unflatten(
                [curOutputs.popleft() for _ in range(len(inputFlattened))],
                inputUnflattened
            )
            yield processOutput(input_, inputUnflattened, outputsUnflattened)


async def runBatchedAsync(
    inputs            : Iterable[Any],
    getInputs         : Callable[[Any], Any],
    processBatch      : Callable[[List[Any]], Any],
    processOutput     : Callable[[Any, Any, Any], Any],
    batchSize         : int,
    verbose           : bool = True,
    noCancel          : bool = False
) -> List[Any]:
    """
    Async counterpart of runBatched. Returns *list* of results.
    """
    results = []
    async for out in runBatchedIteratorAsync(
            inputs       = inputs,
            n            = len(inputs),
            getInputs    = getInputs,
            processBatch = processBatch,
            processOutput= processOutput,
            batchSize    = batchSize,
            verbose      = verbose,
            noCancel     = noCancel):
        results.append(out)
    return results

def roleToSafetyToolingRole(role):
    if role == "user": return MessageRole.user
    if role == "assistant": return MessageRole.assistant
    if role == "system": return MessageRole.system
    raise ValueError(f"Unknown role {role}")
def safetyToolingRoleToRole(safetyToolingRole):
    if safetyToolingRole == MessageRole.user: return "user"
    if safetyToolingRole == MessageRole.assistant: return "assistant"
    if safetyToolingRole == MessageRole.system: return "system"
    raise ValueError(f"Unknown role {safetyToolingRole}")
def messagesToSafetyToolingMessages(messages):
    return [ChatMessage(content=message["content"], role=roleToSafetyToolingRole(message["role"])) for message in messages]
def safetyToolingMessagesToMessages(messages):
    return [{"role": safetyToolingRoleToRole(message.role), "content": message.content} for message in messages]




class FinishedException(Exception):
    pass