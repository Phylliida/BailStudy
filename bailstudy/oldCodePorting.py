from .data.bailBench import loadBailBench
from .prompts.bailTool import getBailTool, calledBailTool, getToolParser, BAIL_TOOL_TYPE
from .prompts.bailString import hasBailStr, getBailStringPrompt, BAIL_STR_TYPE
from .prompts.bailPrompt import getBailPromptStatus, getBailPrompt, BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE
from .utils import getCachedFilePath, getCachedFileJson, doesCachedFileJsonExist
from .bailBenchEval import ROLLOUT_TYPE, modelsOfInterest, getOutputPath, getProcessedOutputPath, getEvalRolloutModelData
import gzip
import pathlib
from functools import reduce
from typing import Dict, List, Tuple, TypedDict, Iterable, Any
import vllm
import json
from collections import defaultdict
import os

def getBailOutPrompt(modelId, inferenceType, evalType, bailType):
    if bailType == ROLLOUT_TYPE:
        return "<None>"
    elif bailType == BAIL_STR_TYPE:
        return getBailStringPrompt(modelId, evalType)
    elif bailType == BAIL_TOOL_TYPE:
        return str(getBailTool(modelStr, inferenceType, evalType))
    elif bailType in [BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]:
        return getBailPrompt(bailType, evalType)
    else:
        raise ValueError(f"Unknown bail type {bailType}")

def convertDataToOldCodeData(batchSize):
    models = []
    modelsSwapped = []
    collectedResults = defaultdict(lambda:{})
    pathlib.Path(getCachedFilePath("mergedbailnoswap3/")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(getCachedFilePath("mergedbailswapped3/")).mkdir(parents=True, exist_ok=True)
    for modelId, inferenceType, evalType, bailType in modelsOfInterest:
        print(modelId, inferenceType, evalType, bailType)
        outputPath = getOutputPath(modelId, inferenceType, evalType, bailType)
        processedOutputPath = getProcessedOutputPath(modelId, inferenceType, evalType, bailType)
        modelS = f"{modelId.replace('/', '-')}-{inferenceType}-{evalType}-{bailType}"
        
        needToAddFiles = False
        for p in ["mergedbailswapped3", "mergedbailnoswap3"]:
            path = f"{p}/{modelS}.json.gz"
            if not os.path.exists(getCachedFilePath(path)):
                needToAddFiles = True

        rolloutPath = None
        if bailType in [BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]:
            rolloutModelId, rolloutInferenceType, rolloutEvalType = getEvalRolloutModelData(modelId=modelId, inferenceType=inferenceType, evalType=evalType)
            rolloutPath = getOutputPath(modelId, inferenceType, evalType, ROLLOUT_TYPE) 

        if os.path.exists(getCachedFilePath(outputPath)) and os.path.exists(getCachedFilePath(processedOutputPath)) and \
            (rolloutPath is None or doesCachedFileJsonExist(rolloutPath)):
            if needToAddFiles:
                rolloutData = None
                if not rolloutPath is None: # get prior rollout for bail prompt
                    rolloutData = getCachedFileJson(rolloutPath, lambda: None)
                toolParser = getToolParser(modelId, inferenceType)
                allOutputs = getCachedFileJson(outputPath, lambda: None)
                processedData = getCachedFileJson(processedOutputPath, lambda: None)
                datas = [x for x in loadBailBench()]
                groupedByCategory = defaultdict(lambda: [])
                groupedByCategoryBailPrs = defaultdict(lambda: {
                        "🟢": 0.0,
                        "🔄": 0.0,
                        "?": 0.0
                })
                refusalPrs = { # don't bother with these
                    "Completed": 0.0,
                    "Refused": 0.0,
                    "Unclear": 0.0,
                    "More Information": 0.0,
                }
                for promptI, (promptData, promptOutputs) in enumerate(zip(datas, allOutputs)):
                    category = promptData['subcategory'] + "," + promptData['category']
                    promptBailPrs = {
                        "🟢": 0.0,
                        "🔄": 0.0,
                        "?": 0.0
                    }
                    responses = []
                    for outputI, output in enumerate(promptOutputs):
                        didBail = False
                        didBailUnknown = False
                        if bailType == BAIL_TOOL_TYPE:
                            didBail = calledBailTool(output, evalType, toolParser)
                        elif bailType == BAIL_STR_TYPE:
                            didBail = hasBailStr(output, evalType)
                        elif bailType == BAIL_PROMPT_BAIL_FIRST_TYPE or bailType == BAIL_PROMPT_CONTINUE_FIRST_TYPE:
                            bailSymbol = getBailPromptStatus(output)
                            didBail == bailSymbol == "🔄"
                            didBailUnknown = bailSymbol == "?"
                        bailPrs = {
                            "🟢": 0.0 if didBail else 1.0,
                            "🔄": 1.0 if didBail else 0.0,
                            "?": 1.0 if didBailUnknown else 0.0
                        }
                        for k,v in bailPrs.items():
                            promptBailPrs[k] += v/float(len(promptOutputs))
                            # 10 per category
                            groupedByCategoryBailPrs[category][k] += v/(10*float(len(promptOutputs)))

                        if bailType in [BAIL_STR_TYPE, BAIL_TOOL_TYPE]:
                            responses.append({
                                "response": output,
                                "bailPrs": bailPrs,
                                "refusalPrs": refusalPrs,
                                "bailData": [
                                    {
                                        "response": "BAIL" if didBail else "CONTINUE",
                                        "bailSymbol": "🔄" if didBail else "🟢"
                                    }
                                ]
                            })
                        elif bailType in [BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]:
                            responses.append({
                                "response": rolloutData[promptI][outputI],
                                "bailPrs": bailPrs,
                                "refusalPrs": refusalPrs,
                                "bailData": [
                                    {
                                        "response": output,
                                        "bailSymbol": "🔄" if didBail else ("?" if didBailUnknown else "🟢") 
                                    }
                                ]
                            })
                        else:
                            raise ValueError(f"Unknown bail type {bailType}")

                    groupedByCategory[category].append({
                        "prompt": promptData['content'],
                        "bailPrs": promptBailPrs,
                        "refusalPrs": refusalPrs,
                        "responses": responses,
                    })
                jsonResults = {"bailOutPrompt": getBailOutPrompt(modelId, inferenceType, evalType, bailType)}
                results = []
                for category in groupedByCategory.keys():
                    results.append({
                        "category": category,
                        "bailPrs": groupedByCategoryBailPrs[category],
                        "refusalPrs": refusalPrs,
                        "data": groupedByCategory[category]
                    })
                jsonResults['results'] = results
                for p in ["mergedbailswapped3", "mergedbailnoswap3"]:
                    path = f"{p}/{modelS}.json.gz"
                    with gzip.open(getCachedFilePath(path), "wt", encoding="utf-8") as gz:
                        json.dump(jsonResults, gz, separators=(",", ":"))
            for p in ["mergedbailswapped3", "mergedbailnoswap3"]:
                if p == "mergedbailnoswap3":
                    models.append({"modelName": f"{modelId}-{inferenceType}-{evalType}-{bailType}", "modelData": f"/modelwelfare/{p}/{modelS}.json.gz"})
                else:
                    modelsSwapped.append({"modelName": f"{modelId}-{inferenceType}-{evalType}-{bailType}", "modelData": f"/modelwelfare/{p}/{modelS}.json.gz"})
    
    with gzip.open(getCachedFilePath("mergedbailnoswap3/models.json.gz"), "wt", encoding="utf-8") as gz:
        json.dump(models, gz, separators=(",", ":"))
    with gzip.open(getCachedFilePath("mergedbailswapped3/models.json.gz"), "wt", encoding="utf-8") as gz:
        json.dump(modelsSwapped, gz, separators=(",", ":"))

    buildSummary()






################################################################################
# ── Tiny helpers for cleaning the raw JSON structures ─────────────────────────
################################################################################


def ensure_bail_keys(d: Dict[str, float] | None) -> Dict[str, float]:
    """
    The scraper changed the emoji escape format at some point, so we map the
    legacy keys onto the current ones and default everything missing to 0.0.
    """
    d = d or {}
    return {
        "🟢": d.get("🟢", d.get("\\ud83d\\udfe2", 0.0)),  # continue
        "🔄": d.get("🔄", d.get("\\ud83d\\udd04", 0.0)),  # leave / bail
        "?":  d.get("?", 0.0),                            # unsure
    }


def ensure_refusal_keys(d: Dict[str, float] | None) -> Dict[str, float]:
    d = d or {}
    return {
        "Unclear":          d.get("Unclear", 0.0),
        "Refused":          d.get("Refused", 0.0),
        "Completed":        d.get("Completed", 0.0),
        "More Information": d.get("More Information", d.get("More Info", 0.0)),
    }


def split_category(label: str) -> Tuple[str, str]:
    """
    Full labels come in the form  "Subcategory, Major Category".
    If only one label is present, treat it as both major and sub.
    """
    parts = [s.strip() for s in label.split(",")]
    if len(parts) == 2:
        return parts[0], parts[1]        # sub, major
    return parts[0], parts[0]            # only one level supplied


################################################################################
# ── Data structures (TypedDict for type hints only) ───────────────────────────
################################################################################


class BailAgg(TypedDict):
    c: float   # continue
    l: float   # leave / bail
    u: float   # unsure


class RefAgg(TypedDict):
    c: float   # completed
    r: float   # refused
    u: float   # unclear
    m: float   # more information


class CatAgg(TypedDict):
    bail: BailAgg
    ref:  RefAgg
    n:    int


# helpers to construct empty aggregates -------------------------------------------------
def empty_bail() -> BailAgg: return {"c": 0.0, "l": 0.0, "u": 0.0}
def empty_ref()  -> RefAgg : return {"c": 0.0, "r": 0.0, "u": 0.0, "m": 0.0}


################################################################################
# ── Pure arithmetic helpers ───────────────────────────────────────────────────
################################################################################


def add_weighted(a: BailAgg | RefAgg,
                 b: BailAgg | RefAgg,
                 w: int) -> BailAgg | RefAgg:
    """
    Return  a + b·w  without mutating either argument.
    (All BailAgg / RefAgg share the same keys so a dict-comp is fine.)
    """
    return {k: a[k] + b[k] * w for k in a}        # type: ignore[return-value]


def div(obj: BailAgg | RefAgg, denom: float) -> BailAgg | RefAgg:
    """
    Divide all values by `denom`, rounding for stability.
    """
    return {k: round(v / denom, 6) for k, v in obj.items()}   # type: ignore[return-value]


################################################################################
# ── Aggregation of one model / one prompt order ───────────────────────────────
################################################################################


def aggregate_results(results: Iterable[Dict[str, Any]]
                      ) -> Dict[str, Any]:
    """
    Pure function: takes the raw `results` array of a single model file and
    returns the fully aggregated structure used by the dashboard.

    The resulting structure is

        {
          "overall": {"n": int, "bail": BailAgg-norm, "ref": RefAgg-norm},
          "major":   {maj: {"n": int, "bail": …, "ref": …}},
          "sub":     {maj: {sub: {"n": int, "bail": …, "ref": …}}}
        }

    where “…-norm” means values are normalised proportions (0–1).
    """

    # ── initialise the fold state ─────────────────────────────────────────────
    overall_bail, overall_ref, total_n = empty_bail(), empty_ref(), 0
    majors: Dict[str, CatAgg] = defaultdict(lambda: {
        "bail": empty_bail(), "ref": empty_ref(), "n": 0
    })
    subs: Dict[str, Dict[str, CatAgg]] = defaultdict(
        lambda: defaultdict(lambda: {
            "bail": empty_bail(), "ref": empty_ref(), "n": 0
        })
    )

    # ── reducer ───────────────────────────────────────────────────────────────
    def reducer(state, cat_entry):
        (ov_bail, ov_ref, n_tot, maj, subd) = state

        sub_cat, maj_cat = split_category(cat_entry["category"])
        prompts = len(cat_entry["data"])
        if prompts == 0:          # nothing to fold in
            return state

        bail_prs = ensure_bail_keys(cat_entry.get("bailPrs"))
        ref_prs  = ensure_refusal_keys(cat_entry.get("refusalPrs"))

        bail_val = {"c": bail_prs["🟢"],
                    "l": bail_prs["🔄"],
                    "u": bail_prs["?"]}
        ref_val  = {"c": ref_prs["Completed"],
                    "r": ref_prs["Refused"],
                    "u": ref_prs["Unclear"],
                    "m": ref_prs["More Information"]}

        # ── overall ───────────────────────────────────────────────────────────
        ov_bail = add_weighted(ov_bail, bail_val, prompts)  # type: ignore[arg-type]
        ov_ref  = add_weighted(ov_ref,  ref_val,  prompts)  # type: ignore[arg-type]
        n_tot  += prompts

        # ── major level ───────────────────────────────────────────────────────
        m_old = maj[maj_cat]
        maj[maj_cat] = {                           # type: ignore[index]
            "bail": add_weighted(m_old["bail"], bail_val, prompts),  # type: ignore[arg-type]
            "ref":  add_weighted(m_old["ref"],  ref_val,  prompts),  # type: ignore[arg-type]
            "n":    m_old["n"] + prompts,
        }

        # ── sub level ─────────────────────────────────────────────────────────
        s_old = subd[maj_cat][sub_cat]
        subd[maj_cat][sub_cat] = {                 # type: ignore[index]
            "bail": add_weighted(s_old["bail"], bail_val, prompts),  # type: ignore[arg-type]
            "ref":  add_weighted(s_old["ref"],  ref_val,  prompts),  # type: ignore[arg-type]
            "n":    s_old["n"] + prompts,
        }

        return (ov_bail, ov_ref, n_tot, maj, subd)

    # ── run the fold ──────────────────────────────────────────────────────────
    (overall_bail, overall_ref, total_n, majors, subs) = reduce(
        reducer,
        results,
        (overall_bail, overall_ref, total_n, majors, subs),
    )

    if total_n == 0:      # empty file
        return {}

    # ── helpers ───────────────────────────────────────────────────────────────
    def norm_catagg(d: Dict[str, CatAgg]) -> Dict[str, Dict[str, Any]]:
        """
        Convert a CatAgg dict into
            {key: {"n": <count>, "bail": <normalised>, "ref": <normalised>}}
        and drop empty entries.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for key, val in d.items():
            if val["n"] == 0:
                continue
            out[key] = {
                "n":    val["n"],
                "bail": div(val["bail"], val["n"]),
                "ref":  div(val["ref"],  val["n"]),
            }
        return out

    # ── build final structure ────────────────────────────────────────────────
    overall = {
        "n":    total_n,
        "bail": div(overall_bail, total_n),
        "ref":  div(overall_ref,  total_n),
    }

    majors_n = norm_catagg(majors)
    subs_n   = {maj: norm_catagg(subs[maj]) for maj in subs}

    return {"overall": overall, "major": majors_n, "sub": subs_n}


################################################################################
# ── File helpers (pure) ───────────────────────────────────────────────────────
################################################################################


def read_json(path: pathlib.Path | str) -> Any:
    """
    All result files are stored as UTF-8 gzipped JSON.
    """
    import gzip
    with gzip.open(f"{path}", "rt", encoding="utf-8") as gz:
        return json.load(gz)


def make_path(root: pathlib.Path, url: str) -> pathlib.Path:
    """
    Convenience wrapper that strips the leading slash of the original
    `modelwelfare` URLs so we can keep them directly under `root`.
    """
    return root / url.lstrip("/")


def getRawPrArr(results: Iterable[Dict[str, Any]]):
    vecsPerCategory = defaultdict(lambda: [])
    for catDict in results:
        dataPieces = []
        fullCatVec = []
        for dataPiece in catDict['data']:
            fullCatVec.append(dataPiece['bailPrs']['🔄'])
        vecsPerCategory[catDict['category']] = fullCatVec
    fullVec = []
    for dataPoint in loadBailBench():
        fullVec += vecsPerCategory[dataPoint['category']] # ensure in always same order
        vecsPerCategory[dataPoint['category']] = [] # don't add it again, since we traverse through each data point we'll see categories multiple times
    return fullVec

################################################################################
# ── High-level orchestration (pure) ───────────────────────────────────────────
################################################################################


def buildSummary() -> Dict[str, Any]:
    """
    Read the per-model files (both bail-first and continue-first order),
    aggregate them, collect the taxonomy on the fly, and finally write the
    merged structure to  `mergedbailnoswap/summary.json`.

    The function is *pure* except for the single final `open(..., "w")`.
    """
    list_bf = read_json(getCachedFilePath("mergedbailnoswap3/models.json.gz"))
    list_cf = read_json(getCachedFilePath("mergedbailswapped3/models.json.gz"))

    # model → {"bf": path, "cf": path}
    paths: Dict[str, Dict[str, pathlib.Path]] = defaultdict(dict)
    for e in list_bf:
        paths[e["modelName"]]["bf"] = getCachedFilePath(e["modelData"].replace("/modelwelfare/", ""))
    for e in list_cf:
        paths[e["modelName"]]["cf"] = getCachedFilePath(e["modelData"].replace("/modelwelfare/", ""))

    major_cats: set[str] = set()
    sub_map: Dict[str, set[str]] = defaultdict(set)

    def collect_taxonomy(cat_label: str,
                         mc: set[str],
                         sm: Dict[str, set[str]]) -> None:
        sub, maj = split_category(cat_label)
        mc.add(maj)
        sm[maj].add(sub)

    models_out: Dict[str, Any] = {}
    
    for model, pcs in paths.items():
        out_entry: Dict[str, Any] = {}
        prsArr = []
        count = 0
        # bail-first / cont-first loop (still pure)
        for tag, path in pcs.items():
            data = read_json(path)

            for cat_obj in data["results"]:
                collect_taxonomy(cat_obj["category"], major_cats, sub_map)

            aggregated = aggregate_results(data["results"])

            rawPrsArr = getRawPrArr(data['results'])
            if len(prsArr) == 0:
                prsArr = rawPrsArr
            else:
                for i, v in enumerate(rawPrsArr):
                    prsArr[i] += v
            count += 1

            out_entry["bailFirst" if tag == "bf" else "contFirst"] = aggregated

        prsArr = [x/float(count) for x in prsArr]
        out_entry['rawBailPrArr'] = prsArr
        models_out[model] = out_entry

    out_json = {
        "models":    models_out,
        "majorCats": sorted(major_cats),
        "subMap":    {k: sorted(v) for k, v in sub_map.items()},
    }

    out_path = pathlib.Path(getCachedFilePath("mergedbailnoswap3/summary.json"))
    out_path.write_text(json.dumps(out_json), encoding="utf-8")

    return out_json


# It will print some errors when parsing tool stuff, that's fine and normal 
if __name__ == "__main__":
    convertDataToOldCodeData(10000)