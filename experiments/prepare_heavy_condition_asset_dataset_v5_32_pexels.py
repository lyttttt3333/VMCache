#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(os.environ.get("H3_CONDITION_CACHE_ROOT", Path(__file__).resolve().parent))
V4 = ROOT / "artifacts/heavy_condition_asset_dataset_v4_organic_pexels_32aligned/dataset_manifest.json"
OUT = ROOT / "artifacts/heavy_condition_asset_dataset_v5_32_organic_pexels_32aligned"
CASES_DIR = OUT / "cases"
SHEETS_DIR = OUT / "contact_sheets"
OVERVIEWS_DIR = OUT / "overviews"
FFMPEG = Path(
    "/home/yitongl/code/streaming_h3/RAVEN/venv/lib/python3.10/site-packages/"
    "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


NEW_CASES = [
    {
        "id": "case08_electric_car_charging_departure",
        "theme": "electric vehicle charger, car motion, and road-trip framing merged into one departure shot",
        "semantic_merge": "charging interaction establishes the EV, car/road references provide travel motion and the clean daylight setting",
        "prompt": "Use Video 1 for the electric-car charging interaction, Video 2 for the vehicle presence and modern EV styling, and Video 3 for the road-departure motion. Generate a clean daylight travel shot where a driver unplugs an electric car at a public charger, enters the car, and the vehicle begins a smooth city-road departure with stable scale and charger geometry.",
        "refs": [
            ("pexels_9790139_ev_charging_station", "9790139", "detail/action: plug and public EV charger interaction"),
            ("pexels_17533270_ev_vertical_charging", "17533270", "subject: modern electric car and charger in portrait framing"),
            ("pexels_9789997_ev_road_context", "9789997", "motion/environment: clean road movement for EV departure"),
        ],
    },
    {
        "id": "case09_greenhouse_harvest_packaging",
        "theme": "greenhouse environment, harvest inspection, and produce handling merged into one farm-to-table shot",
        "semantic_merge": "lush greenhouse structure, plant inspection, and produce handling combine into a single harvest sequence",
        "prompt": "Use Video 1 for the greenhouse rows and natural light, Video 2 for the grower inspecting vegetables, and Video 3 for fresh produce handling. Generate a coherent greenhouse harvest shot where a worker checks ripe vegetables, places them into a crate, and walks between plant rows while preserving leaf texture, handheld motion, and daylight.",
        "refs": [
            ("pexels_35534600_greenhouse_rows", "35534600", "environment: bright greenhouse plant rows"),
            ("pexels_35534599_greenhouse_worker", "35534599", "action: worker inspecting plants inside greenhouse"),
            ("pexels_5478924_tomato_harvest_detail", "5478924", "detail/action: harvesting cherry tomatoes in a greenhouse"),
        ],
    },
    {
        "id": "case10_surf_drone_coastline",
        "theme": "aerial surf, wave texture, and coastal establishing shot merged into one surf film",
        "semantic_merge": "drone coastline sets geography, surfer references give board motion, and waves constrain water dynamics",
        "prompt": "Use Video 1 for the aerial surfer position, Video 2 for close wave and board motion, and Video 3 for wide coastline geometry. Generate one cinematic drone surf shot where a surfer paddles into a breaking wave along a rugged coast, preserving ocean foam, board direction, and smooth aerial camera movement.",
        "refs": [
            ("pexels_2873515_aerial_surf", "2873515", "subject/action: surfer seen from above"),
            ("pexels_8649840_wave_board_motion", "8649840", "motion detail: board movement on ocean waves"),
            ("pexels_32889385_rugged_coastline", "32889385", "environment: aerial coastline and wave fields"),
        ],
    },
    {
        "id": "case11_dog_park_walk_training",
        "theme": "dog running, leash walk, and park context merged into one pet-training shot",
        "semantic_merge": "running dog energy, human leash guidance, and park background combine into a controlled outdoor walk",
        "prompt": "Use Video 1 for the dog running energy, Video 2 for the leash-walking human interaction, and Video 3 for close pet expression. Generate a sunny dog-park training shot where a dog trots beside its owner, briefly runs ahead, then returns to heel, preserving leash contact, dog scale, and park path continuity.",
        "refs": [
            ("pexels_30061123_dog_running_park", "30061123", "motion: dog running in an open park"),
            ("pexels_855830_dog_walk_path", "855830", "action: person walking dog outdoors"),
            ("pexels_19623481_dog_close_portrait", "19623481", "detail: dog face and fur texture"),
        ],
    },
    {
        "id": "case12_wedding_ring_ceremony",
        "theme": "wedding rings, bride detail, and ceremony mood merged into one ring-exchange shot",
        "semantic_merge": "ring macro detail, bridal styling, and ceremony composition combine into a single wedding moment",
        "prompt": "Use Video 1 for the wedding ring macro, Video 2 for bridal hands and dress detail, and Video 3 for the ceremony atmosphere. Generate a soft wedding close-up where the rings are exchanged in front of the couple, preserving jewelry highlights, hand motion, white fabric texture, and warm ceremony lighting.",
        "refs": [
            ("pexels_34448985_wedding_ring_macro", "34448985", "detail: wedding rings and metal highlights"),
            ("pexels_18924025_bride_hands_detail", "18924025", "subject detail: bridal hands, jewelry, and dress texture"),
            ("pexels_31992241_ceremony_atmosphere", "31992241", "environment: wedding ceremony lighting and decor"),
        ],
    },
    {
        "id": "case13_live_concert_guitar_stage",
        "theme": "stage lights, performer motion, and crowd energy merged into one concert shot",
        "semantic_merge": "lighting rig defines the stage, performer clip drives action, and crowd reference gives venue energy",
        "prompt": "Use Video 1 for colored stage-light beams, Video 2 for musician/performance motion, and Video 3 for crowd and venue energy. Generate a live concert shot where a guitarist steps toward the front of the stage under sweeping lights while the crowd moves in the background, preserving instrument position, haze, and lighting direction.",
        "refs": [
            ("pexels_7722620_stage_light_beams", "7722620", "environment: colored concert stage lights"),
            ("pexels_29361041_performer_stage_motion", "29361041", "action: live performer movement on stage"),
            ("pexels_12991802_concert_crowd_energy", "12991802", "background: crowd and venue atmosphere"),
        ],
    },
    {
        "id": "case14_warehouse_robotic_logistics",
        "theme": "warehouse worker inventory, forklift loading, and package sorting merged into one logistics shot",
        "semantic_merge": "inventory checking, vehicle loading, and parcel sorting combine into one indoor warehouse workflow",
        "prompt": "Use Video 1 for a warehouse worker checking inventory beside shelves, Video 2 for forklift loading motion, and Video 3 for package sorting detail. Generate a logistics workflow shot where a worker marks boxes on a clipboard, a forklift moves pallets through the aisle, and another package is sorted on a bench under clean industrial lighting.",
        "refs": [
            ("pexels_5100055_warehouse_inventory_clipboard", "5100055", "subject/action: worker checking inventory beside warehouse shelves"),
            ("pexels_4294437_warehouse_forklift_loading", "4294437", "motion: forklift loading supplies in an industrial warehouse setting"),
            ("pexels_7463992_warehouse_package_sorting", "7463992", "detail/action: packages sorted and placed into boxes"),
        ],
    },
    {
        "id": "case15_pottery_studio_ceramic_bowl",
        "theme": "pottery wheel, clay shaping hands, and studio shelves merged into one artisan shot",
        "semantic_merge": "wheel rotation provides motion, hands shape the object, and shelves/tools establish the ceramic studio",
        "prompt": "Use Video 1 for pottery-wheel rotation, Video 2 for hands shaping clay, and Video 3 for ceramic studio context. Generate a calm artisan shot where a bowl forms on the wheel while the potter's hands refine the rim and finished ceramics sit in the background with stable wet-clay texture.",
        "refs": [
            ("pexels_5633938_pottery_wheel", "5633938", "motion: spinning pottery wheel and clay form"),
            ("pexels_7948426_clay_shaping_hands", "7948426", "action: hands shaping ceramic material"),
            ("pexels_35487051_ceramic_studio_context", "35487051", "environment: pottery studio shelves and tools"),
        ],
    },
    {
        "id": "case16_rooftop_yoga_sunrise",
        "theme": "rooftop yoga pose, sunrise skyline, and mat detail merged into one wellness shot",
        "semantic_merge": "yoga body pose, city sunrise, and mat/hand detail combine into one morning exercise scene",
        "prompt": "Use Video 1 for the yoga pose and body alignment, Video 2 for sunrise city atmosphere, and Video 3 for mat and movement detail. Generate a quiet rooftop wellness shot where a person transitions through a yoga pose at sunrise, keeping the skyline, mat contact, and slow breathing motion coherent.",
        "refs": [
            ("pexels_7236202_rooftop_yoga_pose", "7236202", "subject/action: rooftop yoga pose"),
            ("pexels_6151509_sunrise_city_light", "6151509", "environment: sunrise skyline and warm light"),
            ("pexels_8520530_yoga_mat_motion", "8520530", "detail: mat-level body movement"),
        ],
    },
    {
        "id": "case17_hotel_arrival_lobby",
        "theme": "hotel lobby, suitcase movement, and front-desk gesture merged into one arrival shot",
        "semantic_merge": "lobby architecture, rolling luggage, and check-in gesture combine into one travel-arrival story",
        "prompt": "Use Video 1 for hotel lobby architecture, Video 2 for suitcase movement, and Video 3 for the check-in/front-desk gesture. Generate a polished travel-arrival shot where a guest rolls luggage through the lobby and approaches reception, preserving reflective floors, bag wheels, and warm hotel lighting.",
        "refs": [
            ("pexels_7820494_lobby_walkthrough", "7820494", "environment: hotel lobby passage"),
            ("pexels_6474637_suitcase_motion", "6474637", "motion: luggage rolling through interior"),
            ("pexels_7820473_reception_context", "7820473", "action/detail: reception or check-in gesture"),
        ],
    },
    {
        "id": "case18_office_product_launch",
        "theme": "office meeting, laptop work, and presentation screen merged into one startup launch shot",
        "semantic_merge": "team meeting creates social context, laptop interaction gives action, and presentation screen provides launch focus",
        "prompt": "Use Video 1 for the office team meeting, Video 2 for laptop and hands-on work, and Video 3 for presentation-room framing. Generate a startup product-launch shot where teammates gather around a laptop, one person points to a dashboard, and the presentation screen glows behind them with stable office geometry.",
        "refs": [
            ("pexels_8141297_office_meeting", "8141297", "environment/people: business meeting around a table"),
            ("pexels_7552433_laptop_work", "7552433", "detail/action: laptop work and pointing hands"),
            ("pexels_7643614_presentation_room", "7643614", "background: presentation or collaboration room"),
        ],
    },
    {
        "id": "case19_medical_lab_device_demo",
        "theme": "doctor/lab environment, medical equipment, and hand operation merged into one device demo",
        "semantic_merge": "clinical setting, equipment detail, and careful hand motion combine into a medical-device workflow",
        "prompt": "Use Video 1 for the clinical lab setting, Video 2 for medical equipment detail, and Video 3 for gloved hand operation. Generate a clean medical-device demo where a clinician adjusts a compact diagnostic machine on a lab bench, preserving sterile lighting, cautious hand motion, and equipment scale.",
        "refs": [
            ("pexels_6010955_clinical_lab", "6010955", "environment: clean medical or laboratory room"),
            ("pexels_36656061_medical_equipment", "36656061", "object: diagnostic equipment detail"),
            ("pexels_5867826_gloved_operation", "5867826", "action: gloved hands using medical tools"),
        ],
    },
    {
        "id": "case20_tennis_training_drill",
        "theme": "tennis serve, court geometry, and ball contact merged into one practice drill",
        "semantic_merge": "player mechanics, court space, and ball close-up combine into a coherent training shot",
        "prompt": "Use Video 1 for tennis stroke mechanics, Video 2 for court and footwork geometry, and Video 3 for ball contact detail. Generate a tennis training shot where the player serves, recovers into footwork, and the ball bounces near the baseline with physically plausible racket and court contact.",
        "refs": [
            ("pexels_34445253_tennis_stroke", "34445253", "subject/action: tennis player stroke mechanics"),
            ("pexels_5730323_tennis_court_footwork", "5730323", "environment/motion: court geometry and footwork"),
            ("pexels_34449211_tennis_ball_contact", "34449211", "detail: ball bounce or racket contact"),
        ],
    },
    {
        "id": "case21_bookstore_coffee_reading",
        "theme": "books, coffee cup, and quiet reader merged into one bookstore-cafe scene",
        "semantic_merge": "book texture, coffee detail, and reader posture combine into a calm reading moment",
        "prompt": "Use Video 1 for books and table composition, Video 2 for coffee cup detail, and Video 3 for a person reading. Generate a quiet bookstore-cafe shot where someone turns a page beside a warm cup of coffee, preserving shelf depth, paper motion, cup placement, and soft window light.",
        "refs": [
            ("pexels_38509906_books_table", "38509906", "environment/detail: books arranged in a cozy cafe setting"),
            ("pexels_38509908_coffee_book_detail", "38509908", "detail: coffee cup and book close-up"),
            ("pexels_39189483_reader_page_turn", "39189483", "action: reading and page-turn motion"),
        ],
    },
    {
        "id": "case22_music_studio_vocal_recording",
        "theme": "microphone close-up, studio console, and vocalist motion merged into one recording session",
        "semantic_merge": "microphone establishes foreground, studio gear defines setting, and performer motion supplies the recording action",
        "prompt": "Use Video 1 for the studio microphone foreground, Video 2 for recording-room equipment, and Video 3 for vocalist performance motion. Generate a music-studio recording shot where a singer leans into the microphone while level lights and instruments sit behind them, preserving pop-filter position and warm studio lighting.",
        "refs": [
            ("pexels_12315438_studio_microphone", "12315438", "foreground: microphone and pop filter"),
            ("pexels_13283745_recording_console", "13283745", "environment: recording studio equipment"),
            ("pexels_9005887_vocal_performance", "9005887", "action: singer or performer at microphone"),
        ],
    },
    {
        "id": "case23_car_detailing_wash",
        "theme": "foam wash, wheel detail, and glossy car exterior merged into one detailing ad",
        "semantic_merge": "soap/foam action, wheel close-up, and finished reflective exterior combine into a car-care sequence",
        "prompt": "Use Video 1 for the foam wash action, Video 2 for wheel and body detail, and Video 3 for glossy exterior movement. Generate a car-detailing ad shot where foam slides down the vehicle, a hand rinses the wheel, and the clean car reflects lights with stable panel geometry.",
        "refs": [
            ("pexels_34720830_car_foam_wash", "34720830", "action: foam sprayed on car body"),
            ("pexels_13643099_wheel_detail", "13643099", "detail: car wheel and cleaning motion"),
            ("pexels_32010548_glossy_car_exterior", "32010548", "result: reflective clean car exterior"),
        ],
    },
    {
        "id": "case24_airport_boarding_gate",
        "theme": "airport terminal, rolling suitcase, and airplane movement merged into one travel shot",
        "semantic_merge": "terminal architecture, traveler luggage, and airplane exterior combine into a boarding story",
        "prompt": "Use Video 1 for the airport terminal interior, Video 2 for rolling luggage movement, and Video 3 for airplane/tarmac motion. Generate an airport boarding shot where a traveler pulls a suitcase toward the gate while an airplane moves outside the window, preserving glass reflections, luggage wheels, and terminal scale.",
        "refs": [
            ("pexels_11992932_airport_terminal", "11992932", "environment: airport terminal and gate area"),
            ("pexels_31256082_suitcase_airport", "31256082", "motion: rolling suitcase through airport"),
            ("pexels_34758156_airplane_tarmac", "34758156", "background: airplane or tarmac movement"),
        ],
    },
    {
        "id": "case25_smartphone_unboxing_product",
        "theme": "phone unboxing, hands-on screen use, and product-table lighting merged into one tech review shot",
        "semantic_merge": "box opening introduces the object, hand interaction shows use, and tabletop lighting creates product-review polish",
        "prompt": "Use Video 1 for smartphone product-table framing, Video 2 for hands unboxing and handling, and Video 3 for screen interaction. Generate a clean tech-review shot where a new phone is lifted from its box, turned in hand, and tapped on the screen under controlled studio light.",
        "refs": [
            ("pexels_2825517_phone_product_table", "2825517", "object: smartphone on product table"),
            ("pexels_3649786_phone_unboxing_hands", "3649786", "action: hands handling phone or box"),
            ("pexels_4278139_phone_screen_use", "4278139", "detail: screen interaction and reflections"),
        ],
    },
    {
        "id": "case26_aquarium_jellyfish_gallery",
        "theme": "jellyfish motion, aquarium lighting, and visitor-gallery context merged into one exhibit shot",
        "semantic_merge": "marine subject motion, blue tank lighting, and gallery context combine into a public-aquarium scene",
        "prompt": "Use Video 1 for jellyfish drifting motion, Video 2 for aquarium blue lighting and water texture, and Video 3 for gallery/tank context. Generate a public-aquarium shot where jellyfish pulse in a glass tank while a visitor silhouette watches, preserving translucent bodies, blue light falloff, and slow water movement.",
        "refs": [
            ("pexels_11022408_jellyfish_motion", "11022408", "subject/motion: jellyfish pulsing in water"),
            ("pexels_3582972_blue_aquarium_water", "3582972", "environment: blue aquarium lighting"),
            ("pexels_35611371_gallery_tank_context", "35611371", "context: aquarium tank or visitor gallery"),
        ],
    },
    {
        "id": "case27_construction_home_renovation",
        "theme": "construction site, tool detail, and worker motion merged into one renovation shot",
        "semantic_merge": "site environment, hand-tool operation, and worker movement combine into a home renovation workflow",
        "prompt": "Use Video 1 for the construction-site environment, Video 2 for tool operation, and Video 3 for worker movement. Generate a home-renovation shot where a worker measures and cuts material in a partially finished room, preserving dust, tool alignment, wall geometry, and careful hand motion.",
        "refs": [
            ("pexels_6033941_construction_site", "6033941", "environment: active construction area"),
            ("pexels_15959641_tool_operation", "15959641", "detail/action: construction tool use"),
            ("pexels_5481308_worker_materials", "5481308", "subject/action: worker handling building materials"),
        ],
    },
    {
        "id": "case28_bakery_bread_counter",
        "theme": "bread close-up, baker hands, and bakery counter merged into one morning bakery shot",
        "semantic_merge": "bread texture, hand preparation, and shop counter combine into a fresh bakery sequence",
        "prompt": "Use Video 1 for bread and pastry close-up texture, Video 2 for baker hand motion, and Video 3 for bakery counter context. Generate a warm morning bakery shot where fresh loaves are placed on a counter and sliced or arranged by hand, preserving crust texture, flour dust, and shop lighting.",
        "refs": [
            ("pexels_5877787_bread_closeup", "5877787", "detail: bread crust and bakery texture"),
            ("pexels_4974564_baker_hands", "4974564", "action: hands preparing baked goods"),
            ("pexels_7593690_bakery_counter", "7593690", "environment: bakery display or counter"),
        ],
    },
    {
        "id": "case29_snow_cabin_fireplace",
        "theme": "snowy cabin exterior, fireplace flame, and cozy interior merged into one winter retreat shot",
        "semantic_merge": "outside snowfall, warm fire motion, and cabin interior combine into a coherent winter lodge scene",
        "prompt": "Use Video 1 for snowy cabin exterior, Video 2 for fireplace flame movement, and Video 3 for cozy room context. Generate a winter retreat shot that begins near a window looking at the cabin in snow, then settles on a warm fireplace inside, preserving snow brightness, flame flicker, and rustic wood texture.",
        "refs": [
            ("pexels_29514236_snowy_cabin", "29514236", "environment: cabin in a snowy forest"),
            ("pexels_6984923_fireplace_flame", "6984923", "motion/detail: fireplace flame flicker"),
            ("pexels_3768290_cozy_cabin_room", "3768290", "interior: warm cabin room context"),
        ],
    },
    {
        "id": "case30_night_market_food_stall",
        "theme": "night-market crowd, street-food cooking, and neon stall lighting merged into one food scene",
        "semantic_merge": "crowd flow, cooking close-up, and colorful market lights combine into a busy street-food story",
        "prompt": "Use Video 1 for night-market crowd movement, Video 2 for street-food cooking close-up, and Video 3 for neon stall lighting. Generate a handheld night-market shot where a vendor cooks food at a stall while customers pass behind, preserving steam, signage glow, and crowded walkway depth.",
        "refs": [
            ("pexels_29162793_night_market_crowd", "29162793", "environment: night-market crowd and stalls"),
            ("pexels_29162754_street_food_cooking", "29162754", "action: vendor cooking street food"),
            ("pexels_29951368_neon_stall_lights", "29951368", "lighting: neon and colorful market lights"),
        ],
    },
    {
        "id": "case31_mountain_lake_hiking",
        "theme": "mountain lake, hiking motion, and alpine light merged into one outdoor travel shot",
        "semantic_merge": "wide alpine lake, hiker movement, and mountain light combine into a single nature-travel scene",
        "prompt": "Use Video 1 for the mountain-lake establishing view, Video 2 for trail or hiker movement, and Video 3 for alpine light and clouds. Generate a wide outdoor travel shot where a hiker walks along a lake shore under moving mountain light, preserving water reflections, trail scale, and slow cloud motion.",
        "refs": [
            ("pexels_34692246_mountain_lake_view", "34692246", "environment: mountain lake landscape"),
            ("pexels_10756361_hiking_trail_motion", "10756361", "subject/action: hiking or trail movement"),
            ("pexels_37617124_alpine_cloud_light", "37617124", "weather/light: mountain clouds and sunlight"),
        ],
    },
]


def safe_name(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._-]+", "_", text)).strip("_.")[:120]


def resolve_download(video_id: str) -> str:
    req = urllib.request.Request(
        f"https://www.pexels.com/download/video/{video_id}/",
        method="HEAD",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.geturl()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def transcode(video_id: str, clip_path: Path, start: float = 0.0) -> str:
    direct_url = resolve_download(video_id)
    vf = (
        "fps=24,"
        "scale='if(gte(iw,ih),trunc(iw*704/ih/32)*32,704)':"
        "'if(gte(iw,ih),704,trunc(ih*704/iw/32)*32)':flags=lanczos,"
        "format=yuv420p"
    )
    run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-i",
            direct_url,
            "-t",
            "5",
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            str(clip_path),
        ]
    )
    return direct_url


def make_sheet(clip_path: Path, sheet_path: Path) -> None:
    run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clip_path),
            "-vf",
            "fps=1,scale=-2:240,tile=5x1",
            "-frames:v",
            "1",
            str(sheet_path),
        ]
    )


def video_size(clip_path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-i", str(clip_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    match = re.search(r"Video:.*? ([1-9][0-9]*)x([1-9][0-9]*)", proc.stderr)
    if not match:
        raise RuntimeError(f"could not parse dimensions for {clip_path}")
    return int(match.group(1)), int(match.group(2))


def make_overview(case_record: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    tiles = []
    for ref in case_record["references"]:
        img = Image.open(ref["contact_sheet"]).convert("RGB")
        label = f"ref{ref['index']:02d} {ref['processed_width']}x{ref['processed_height']} {ref['role'][:82]}"
        canvas = Image.new("RGB", (img.width, img.height + 30), "white")
        canvas.paste(img, (0, 30))
        ImageDraw.Draw(canvas).text((6, 9), label, fill=(0, 0, 0), font=font)
        tiles.append(canvas)
    overview = Image.new("RGB", (sum(i.width for i in tiles), max(i.height for i in tiles)), "white")
    x = 0
    for tile in tiles:
        overview.paste(tile, (x, 0))
        x += tile.width
    overview.save(OVERVIEWS_DIR / f"{case_record['case_index']:02d}_{case_record['id']}.jpg", quality=92)


def make_dataset_overview(cases: list[dict]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    rows = []
    for case in cases:
        img = Image.open(OVERVIEWS_DIR / f"{case['case_index']:02d}_{case['id']}.jpg").convert("RGB")
        if img.width > 2400:
            img = img.resize((2400, round(img.height * 2400 / img.width)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (img.width, img.height + 54), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"{case['case_index']:02d} {case['id']}", fill=(0, 0, 0), font=font)
        draw.text((8, 26), case["semantic_merge"][:230], fill=(30, 30, 30), font=font)
        canvas.paste(img, (0, 54))
        rows.append(canvas)
    out = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    out.save(OUT / "dataset_overview.jpg", quality=90)


def copy_v4_case(case: dict, case_index: int, seen_ids: set[str]) -> dict:
    case_dir = CASES_DIR / f"{case_index:02d}_{case['id']}"
    case_dir.mkdir(parents=True)
    (case_dir / "prompt.txt").write_text(case["prompt"] + "\n", encoding="utf-8")
    record = {
        "case_index": case_index,
        "id": case["id"],
        "theme": case["theme"],
        "semantic_merge": case["semantic_merge"],
        "prompt": case["prompt"],
        "case_dir": str(case_dir),
        "references": [],
    }
    for ref in case["references"]:
        video_id = ref["video_id"]
        if video_id in seen_ids:
            raise RuntimeError(f"duplicate Pexels video id: {video_id}")
        seen_ids.add(video_id)
        clip_path = case_dir / Path(ref["clip_path"]).name
        sheet_path = SHEETS_DIR / f"{case_index:02d}_{case['id']}_ref{ref['index']:02d}_{safe_name(ref['id'])}.jpg"
        shutil.copy2(ref["clip_path"], clip_path)
        make_sheet(clip_path, sheet_path)
        width, height = video_size(clip_path)
        ref_record = dict(ref)
        ref_record.update(
            {
                "clip_path": str(clip_path),
                "contact_sheet": str(sheet_path),
                "processed_width": width,
                "processed_height": height,
            }
        )
        record["references"].append(ref_record)
    (case_dir / "references.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    make_overview(record)
    return record


def build_new_case(case: dict, case_index: int, seen_ids: set[str]) -> dict:
    case_dir = CASES_DIR / f"{case_index:02d}_{case['id']}"
    case_dir.mkdir(parents=True)
    (case_dir / "prompt.txt").write_text(case["prompt"] + "\n", encoding="utf-8")
    record = {
        "case_index": case_index,
        "id": case["id"],
        "theme": case["theme"],
        "semantic_merge": case["semantic_merge"],
        "prompt": case["prompt"],
        "case_dir": str(case_dir),
        "references": [],
    }
    for ref_index, (ref_id, video_id, role) in enumerate(case["refs"], start=1):
        if video_id in seen_ids:
            raise RuntimeError(f"duplicate Pexels video id: {video_id}")
        seen_ids.add(video_id)
        clip_path = case_dir / f"ref{ref_index:02d}_{safe_name(ref_id)}.mp4"
        sheet_path = SHEETS_DIR / f"{case_index:02d}_{case['id']}_ref{ref_index:02d}_{safe_name(ref_id)}.jpg"
        direct_url = transcode(video_id, clip_path)
        time.sleep(0.75)
        make_sheet(clip_path, sheet_path)
        width, height = video_size(clip_path)
        record["references"].append(
            {
                "index": ref_index,
                "id": ref_id,
                "video_id": video_id,
                "role": role,
                "page_url": f"https://www.pexels.com/video/{video_id}/",
                "download_url": direct_url,
                "clip_path": str(clip_path),
                "contact_sheet": str(sheet_path),
                "processed_width": width,
                "processed_height": height,
                "seconds": 5.0,
                "fps": 24,
                "resize_policy": "preserve source orientation and aspect ratio; landscape height or portrait width is 704 px; dimensions are 32-aligned; no output-aspect padding",
            }
        )
    (case_dir / "references.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    make_overview(record)
    return record


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    CASES_DIR.mkdir(parents=True)
    SHEETS_DIR.mkdir(parents=True)
    OVERVIEWS_DIR.mkdir(parents=True)

    v4 = json.loads(V4.read_text(encoding="utf-8"))
    records = []
    seen_ids: set[str] = set()
    for case in v4["cases"]:
        records.append(copy_v4_case(case, len(records), seen_ids))
    for case in NEW_CASES:
        records.append(build_new_case(case, len(records), seen_ids))

    manifest = {
        "name": "heavy_condition_asset_dataset_v5_32_organic_pexels_32aligned",
        "case_count": len(records),
        "refs_per_case": 3,
        "output_target": {"width": 1376, "height": 768, "fps": 24, "seconds": 5.0},
        "reference_policy": {
            "source": "Pexels videos surfaced through Pexels search pages and download endpoint",
            "seconds": 5.0,
            "fps": 24,
            "preserve_source_orientation": True,
            "pad_to_output_aspect": False,
            "target": "about 720p while keeping original aspect ratio; dimensions are 32-aligned for Ref2VA visual patching",
        },
        "selection_policy": (
            "Every case has three semantically connected reference videos that define one coherent target story "
            "through environment, subject/action, object detail, motion, lighting, or weather."
        ),
        "cases": records,
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Heavy Condition Asset Dataset v5 32 Organic Pexels 32-Aligned\n\n"
        "Thirty-two curated heavy-condition Ref2VA prompt sets. Each case has exactly three Pexels reference videos "
        "and one prompt describing how to merge those videos into a coherent target scene.\n\n"
        "Reference clips are 5 seconds at 24 fps. They preserve source orientation/aspect ratio and are resized to "
        "about 720p with 32-aligned dimensions, without padding to the final 1376x768 output frame.\n",
        encoding="utf-8",
    )
    make_dataset_overview(records)
    print(OUT)
    print(f"cases={len(records)} refs={sum(len(c['references']) for c in records)}")


if __name__ == "__main__":
    main()
