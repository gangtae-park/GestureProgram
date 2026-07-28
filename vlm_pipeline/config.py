
import os

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PIPELINE_DIR)


# =================== UDP / Networking ===================
HOST = "0.0.0.0"
PORT = 5005                              # receive Unity packets

UNITY_RESULT_PORT = 5006                 # Unity listens for VLM results on this port
UNITY_HOST_OVERRIDE = None               # None = auto from inbound traffic
VLM_PACKET_PREFIX = "VLM_RESULT"         # Unity-side filter / matching prefix


# =================== Display / Canvas ===================
LIVE_WINDOW = "Live Stream"          # header label drawn on the left pane
TARGET_WINDOW = "Target Result"      # header label drawn on the right pane
COMBINED_WINDOW = "MacProgram"       # single OS window that holds both panes
CANVAS_W, CANVAS_H = 1100, 1000
STREAM_W, STREAM_H = 1100, 1000
# Header bar height reserved at the top of each pane for the label -- keeps the
# label text out of the actual canvas so gaze/target overlays stay unobstructed.
PANE_HEADER_H = 32

POINT_RADIUS = 8
POINT_COLOR = (0, 0, 255)                # current gaze marker
TRAIL_COLOR = (0, 255, 255)              # in-progress gesture trail / gaze bbox
TARGET_COLOR = (0, 255, 0)               # YOLO target highlight
BG_COLOR = (30, 30, 30)


# =================== ADB Stream ===================
ADB_CMD = ["adb", "exec-out", "screenrecord", "--output-format=h264", "-"]


def build_ffmpeg_cmd(width: int, height: int):
    return [
        "ffmpeg",
        "-loglevel", "error",
        "-i", "-",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}",
        "-",
    ]


# =================== Gaze / Gesture timing ===================
# The adb screenrecord pipeline lags the UDP gaze/pose stream by ~8 frames
# (~267ms @30fps, measured with pilot_data debug frames). Gaze that gets
# MAPPED ONTO THE STREAMED SCREEN (live dot, gesture trail -> target bbox)
# is therefore taken from this long ago, so it lines up with what the frame
# actually shows. Gesture/pinch event flags are NOT delayed.
GAZE_SCREEN_DELAY_S = 8.0 / 30.0

# Head-rotation compensated targeting: freeze the frame at gesture START and
# re-project the whole gesture's gaze trail into that frame's camera pose
# (see headcomp.py). Requires the 11-field GAZE packet (head quaternion) from
# Unity; old 7-field senders automatically fall back to the legacy behaviour
# (END-frame capture, uncompensated trail).
ENABLE_HEAD_COMPENSATION = True
HEADCOMP_FX, HEADCOMP_FY = 729.0, -665.0
HEADCOMP_CX, HEADCOMP_CY = 575.0, 501.0

GAZE_BBOX_PADDING = 10

# Compare fixation clustering: a C->A gaze sweep drags points across whatever
# sits between the two targets, and a single trail-wide bbox would credit that
# middle object with a big IoU. Compare therefore clusters the trail into
# dwell groups (sequential points within RADIUS of the running centroid, with
# revisit merging), DROPS transit points, and matches each of the two largest
# clusters to its own YOLO box.
COMPARE_CLUSTER_RADIUS_PX = 80
COMPARE_CLUSTER_MIN_POINTS = 3
MIN_GAZE_POINTS_FOR_TARGET = 3
CAPTURE_DELAY_AFTER_END = 0.0

TARGET_SCORE_IOU_WEIGHT = 0.999
TARGET_MIN_OVERLAP = 0.03
TARGET_CROP_PAD_RATIO = 0.10             # 10% padding around target bbox for VLM crop

# Compare: number of objects to target, and a distinct overlay colour per slot.
COMPARE_TOP_N = 2
COMPARE_TARGET_COLORS = [(0, 255, 0), (0, 165, 255)]   # BGR: #1 green, #2 orange


# =================== Calibration (gaze_dir -> norm_xy and back) ===================
RIDGE_MODEL_PATH = "calibration_ridge_model.json"
RIDGE_MODEL_ABS_PATH = os.path.join(_PARENT_DIR, RIDGE_MODEL_PATH)


# =================== YOLO ===================
SEG_CONF = 0.15
SEG_IOU_THRESH = 0.50
# YOLOE (open-vocabulary): detects only the classes named in
# SEG_PROMPT_CLASSES below. To roll back to the closed-set COCO model,
# set SEG_MODEL_PATH back to "yolov8s-seg.pt" (prompts are then ignored).
SEG_MODEL_PATH = "yoloe-11s-seg.pt"
SEG_PROMPT_CLASSES = [
    "canned food",
    "tin can",
    "bottle",
    "cup",
    "document",
    "book",
    "houseplant",
]
MAX_SEGMENTS_TO_RENDER = 15         # Cap drawn boxes only; scoring still uses all


# =================== CLIP + object DB ===================
# EVA-02-L/14 at 336px. Drop-in replacement for the older ViT-B/32; clearer
# score separation on the closed 3-object set. First load auto-downloads
# ~430 MB from HuggingFace.
CLIP_MODEL_NAME = "EVA02-L-14-336"
CLIP_PRETRAINED = "merged2b_s6b_b61k"

# Object DB lives in MacProgram/object_db/.
#   objects.json         metadata + per-object info text
#   images/<obj_id>/*    reference images for each object
#   embeddings.npz       cached CLIP embeddings, regenerated when images change
#                        OR when CLIP_MODEL_NAME/CLIP_PRETRAINED change.
OBJECT_DB_DIR = os.path.join(_PARENT_DIR, "object_db")
OBJECT_DB_JSON = os.path.join(OBJECT_DB_DIR, "objects.json")
OBJECT_DB_IMAGES_DIR = os.path.join(OBJECT_DB_DIR, "images")
OBJECT_DB_EMBEDDINGS_PATH = os.path.join(OBJECT_DB_DIR, "embeddings.npz")

# EVA-02 produces a tighter score distribution than ViT-B/32 -- matched pairs
# tend to land in 0.55-0.75 (with the embedding-space cosine being a bit lower
# than ViT-B/32 numerically). Keep masked_crop off per the reference paper.
CLIP_MATCH_MIN_SCORE = 0.55
CLIP_USE_MASKED_CROP = True


# =================== VLM (OpenAI GPT) ===================
OPENAI_MODEL = "gpt-5"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
VLM_OUTPUT_DIR = os.path.join(_PARENT_DIR, "vlm_outputs")
VLM_REQUEST_TIMEOUT_SEC = 30
VLM_MAX_OUTPUT_TOKENS = 2000
VLM_REASONING_EFFORT = "low"

# Voice Input mode: latency matters (user is waiting mid-conversation), so it
# uses a fast non-reasoning multimodal model and downscales the frame before
# the GPT call (fewer image tokens; YOLO/CLIP still run on the full frame).
VOICE_OPENAI_MODEL = "gpt-4o"
VOICE_IMAGE_MAX_SIDE = 768


# =================== User study: response-latency equalisation ===========
# UI (XR-Objects) answers from a prepaid DB lookup in ~10 ms while Gesture
# pays YOLO + CLIP (~1.3 s) at query time. So that response latency does not
# confound the UX comparison, the UI action path HOLDS its result until this
# many seconds have passed since the menu click, matching Gesture's typical
# system time. Set 0 to disable.
UI_RESULT_DELAY_S = 1.0


# =================== OCR / Translate ===================
OCR_ROI_PAD_PX = 240
OCR_ROI_MIN_SIDE = 480
OCR_PARAGRAPH = True
OCR_MAX_SIDE = 960
OCR_MIN_CONF = 0.30


# =================== PROMPTS ===================

ASK_REFERENCE_PROMPT = """\
You are answering a user's open-ended question about a real-world object they pointed at in XR.
The image is a tight crop around the object. The object has already been identified for you
in a "DB info" block below -- treat that as authoritative ground truth and only fall back to
the image for things the DB does not cover (colour, condition, position, etc.). If the DB and
the image still aren't enough (e.g. the question is about specs, availability, recall status,
or anything outside the DB), you may use the web_search tool to fetch external info.

Output format (STRICT):
- Respond with PLAIN TEXT only -- the answer itself, nothing else.
- Do NOT wrap the answer in JSON, quotes, code fences, or any other envelope.
- Do NOT prefix with labels like "answer:" or the object name.
- The stream is rendered live in an XR card as tokens arrive, so any wrapper
  characters would appear in the UI.

Style:
- Natural conversational Korean, complete sentence(s).
- Concise -- one or two sentences is usually enough. Longer only when the
  user's question genuinely calls for detail.
- If the question can't be answered even with web search, say so plainly
  instead of guessing.
"""

# -------- Voice command: GazePointAR-style single multimodal call --------
# Voice Input mode: question end -> scene + gaze captured -> YOLO picks the
# segment at the gaze pixel and CLIP matches it against object_db -> ONE GPT
# call carrying the frame, the gaze pixel ({gaze_info}), the DB lookup result
# ({db_info}, authoritative ground truth when matched), and the verbatim
# transcript. The model classifies the 7-way intent and answers grounded in
# the DB fields; Unity's ResultCardSpawner spawns the card for that intent.
# (A lookup∥GPT parallel variant was tried and reverted: without {db_info}
# in the prompt, Ask answers lose their grounding.)
#
# The intent names MUST match the strings that Unity's ResultCardSpawner
# switches on (see ResultCardSpawner.HandleResult in Assets/Scripts/):
#   "Search", "Ask", "Translate", "Compare", "Anchor", "Save", "Capture".
#
# Filled via str.replace (NOT str.format -- the JSON braces below would
# break format()): {transcript}, {gaze_info}, {db_info}.
VOICE_COMMAND_PROMPT = """\
You are a context-aware voice assistant for a user wearing an XR headset. The
image below is a snapshot of the user's real-world field of view (passthrough
camera of headset) captured the moment their spoken question ended.
Treat pronouns in the query (this, that, here, there, it, they, 이것, 저것,
여기, 저기 등) as pointers to something visible in that image.

===== 1. User query =====
The user asked: "{transcript}"

===== 2. Where the user was LOOKING when the question ended (gaze target) =====
{gaze_info}

===== 3. Identified object at the gaze (database lookup) =====
{db_info}

===== 4. Other objects in view (peripheral context) =====
Everything else visible in the frame (background objects, text)
may still matter for questions like "what else is here?" or
"which of these ...?", but weight it below the gaze target when
resolving a specific referent.

===== 5. Classify the user's intent =====
Decide which ONE of the seven canonical actions the user wants done with the
resolved referent. This decides which result card the headset shows, so the
intent must be the EXACT canonical English string:
1. "Search" -- factual info / description of the target. "what is this?",
   "tell me about this", "이거 뭐야?", "설명해줘".
2. "Translate" -- translate text visible on/near the target. "translate
   this", "이거 번역해줘/해석해줘".
3. "Compare" -- compare two visible objects. "which one is better?",
   "이거랑 저거 비교해줘".
4. "Anchor" -- drop a spatial pin on the target to log the position. "pin this
   here", "여기 표시해줘", "위치 저장해".
5. "Save" -- attach or write a note to the target. "note this", "이거 메모해줘".
6. "Capture" -- photograph / capture the target. "take a picture of this",
   "찍어줘", "사진 저장".
7. "Ask" -- ANY open-ended question that does not clearly fit the six above.
   This is the fallback bucket: when in doubt, choose "Ask".

===== 6. Output format (STRICT) =====
Respond with EXACTLY one JSON object, no prose before or after:

{
  "intent":       "Search" | "Translate" | "Compare" | "Anchor" | "Save" | "Capture" | "Ask",
  "name":         "<short label for the resolved target or task>",
  "answer":       "<ONE natural sentence that directly answers the user, followed by a short justification clause>",
  "note_content": "<Save intent only: the note body spoken inline (wrapping verbs like 'save'/'메모해줘' and framing quotes like '라고'/'that says' stripped, original language preserved); empty string otherwise or when Save has no clear note body>"
}

===== 7. Answering rules =====
- Answer in the SAME language the user spoke (Korean transcript -> Korean
  answer; English transcript -> English answer). "intent" stays the exact
  canonical English string regardless of language.
- When section 3 contains a database match, treat its fields as
  AUTHORITATIVE ground truth about the gaze target: use the DB name as
  "name", answer factual questions (price, capacity, origin, printed
  text, translation) from the DB fields FIRST, and only fall back to the
  image for things the DB does not cover. Do NOT contradict the DB.
- When section 3 reports no match, identify the referent from the image
  alone, exactly as before.
- Keep the tone natural and conversational, like answering a curious friend.
- The "answer" field must be a single sentence. Include a short "because ..."
  or "-- <reason>" clause so the user understands why.
- Ground your answer in what is actually visible AT OR NEAR the gaze pixel
  first. Do NOT invent objects that are not in the frame.
- If the object at the gaze target is unclear, admit it in "answer" and
  describe what IS at that pixel neighbourhood so the user can tell whether
  the tracker mis-aimed.
- Even with missing info or an ambiguous referent, DO NOT refuse. Give your
  best estimate or a range.
- If the query has no pronoun and does not refer to the visible scene at
  all (e.g. a general knowledge question), still answer using the same
  format and classify the intent from the wording alone (usually "Ask").
"""
