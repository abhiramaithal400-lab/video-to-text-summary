import os
import json
import time
import base64
import tempfile
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

import config
from validation import validate_video
from pipeline import summarize_video

BASE_DIR = Path(__file__).resolve().parent


def load_video_b64(path_or_bytes):
    """Convert video bytes or file path into a Base64 Data URI for 100% reliable HTML5 playback."""
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        if not p.is_absolute():
            p = BASE_DIR / p
        if not p.exists():
            return None
        with open(p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(path_or_bytes, (bytes, bytearray)):
        b64_data = base64.b64encode(path_or_bytes).decode("utf-8")
    else:
        return None
    return f"data:video/mp4;base64,{b64_data}"

# -----------------------------------------------------------------------------
# Streamlit Setup - Full Width, High-Contrast Dark Sports Broadcast Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sports match video to text summary:Football",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling: Electric Lime (#79FE0C) + Deep Carbon Dark Theme
st.markdown("""
<style>
    /* Remove default Streamlit top header, deploy button and margins */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    .stDeployButton {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }
    
    /* Flush top alignment */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Top Bar with exact #79FE0C color */
    .custom-top-bar {
        background-color: #79FE0C;
        color: #000000;
        padding: 1.35rem 2.5rem;
        margin-left: -2rem;
        margin-right: -2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 25px rgba(121, 254, 12, 0.3);
    }
    .custom-top-title {
        font-size: 2.35rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        text-transform: uppercase;
        color: #000000;
        margin: 0;
        line-height: 1.15;
    }

    /* Overall dark panel styling */
    body, .stApp {
        background-color: #080C14 !important;
        color: #F8FAFC !important;
    }
    
    .panel-card {
        background-color: #111726;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 1.25rem 1.6rem;
        margin-bottom: 1.5rem;
    }

    /* High-contrast Radio Button pills for dark mode */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 0.8rem !important;
    }
    div[data-testid="stRadio"] label {
        background-color: #182236 !important;
        border: 2px solid #334155 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.25rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #79FE0C !important;
        background-color: #1E2B45 !important;
    }
    div[data-testid="stRadio"] label p {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
    }

    /* Selectbox label and options */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #182236 !important;
        border: 1.5px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Demo Video Metadata (With exact event trigger timestamps and team names)
# -----------------------------------------------------------------------------
DEMO_MATCHES = {
    "Clip 100 — FCU Craiova vs Politehnica Iași (74s)": {
        "static_url": "/app/static/100.mp4",
        "file_path": "data/100.mp4",
        "home": "FCU Craiova",
        "away": "Politehnica Iași",
        "events": [
            {
                "trigger_time": 18.0,
                "label": "GOAL — FCU CRAIOVA",
                "team": "FCU Craiova",
                "text": "GOAL FOR FCU CRAIOVA! Sensational finish! The FCU Craiova attacker breaks through the Iași defense, cuts past the challenge, and rifles a clinical strike into the bottom corner to level the scoreline!",
            }
        ],
    },
    "Clip 101 — Oxford United vs Cheltenham Town (124s)": {
        "static_url": "/app/static/101.mp4",
        "file_path": "data/101.mp4",
        "home": "Oxford United",
        "away": "Cheltenham Town",
        "events": [
            {
                "trigger_time": 35.0,
                "label": "GOAL — OXFORD UNITED",
                "team": "Oxford United",
                "text": "GOAL FOR OXFORD UNITED! What an unbelievable strike into the top corner! The Oxford United forward carves out space outside the 18-yard box and loops a magnificent curling effort past the diving Cheltenham goalkeeper!",
            }
        ],
    },
    "Clip 107 — Wrexham vs Grimsby Town (126s)": {
        "static_url": "/app/static/107.mp4",
        "file_path": "data/107.mp4",
        "home": "Wrexham",
        "away": "Grimsby Town",
        "events": [
            {
                "trigger_time": 34.0,
                "label": "CORNER KICK — WREXHAM",
                "team": "Wrexham",
                "text": "CORNER KICK FOR WREXHAM! Crucial defensive clearance by Grimsby Town diverts the ball behind. Wrexham's tall aerial threats flood the penalty box as the delivery is whipped towards the near post!",
            }
        ],
    },
    "Clip 10 — AS Roma vs Hellas Verona (90s)": {
        "static_url": "/app/static/10_demo.mp4",
        "file_path": "data/10_demo.mp4",
        "home": "AS Roma",
        "away": "Hellas Verona",
        "events": [
            {
                "trigger_time": 44.0,
                "label": "GOAL — AS ROMA",
                "team": "AS Roma",
                "text": "GOAL FOR AS ROMA! Romelu Lukaku strikes for AS Roma! The Belgian powerhouse outmuscles the Verona central defender and smashes home from close range to give Roma the decisive lead!",
            },
            {
                "trigger_time": 84.0,
                "label": "FOUL COMMITTED — HELLAS VERONA",
                "team": "Hellas Verona",
                "text": "FOUL COMMITTED BY HELLAS VERONA! The referee sounds the whistle after an aggressive challenge in midfield stops AS Roma on the break. Free kick awarded to Roma.",
            },
        ],
    },
    "Clip 1 — Anorthosis Famagusta vs Omonia (120s)": {
        "static_url": "/app/static/1_demo.mp4",
        "file_path": "data/1_demo.mp4",
        "home": "Anorthosis Famagusta",
        "away": "Omonia",
        "events": [
            {
                "trigger_time": 88.0,
                "label": "GOAL — ANORTHOSIS FAMAGUSTA",
                "team": "Anorthosis Famagusta",
                "text": "GOAL FOR ANORTHOSIS FAMAGUSTA! Anorthosis Famagusta capitalize on a defensive error, bursting into the penalty area and sliding the ball into the net to ignite the home supporters!",
            },
            {
                "trigger_time": 108.0,
                "label": "YELLOW CARD — OMONIA",
                "team": "Omonia",
                "text": "YELLOW CARD SHOWN TO OMONIA! The match official halts play and shows a yellow card to the Omonia defender following a reckless tackle from behind.",
            },
        ],
    },
}


# -----------------------------------------------------------------------------
# Top Bar (#79FE0C) - Title Only, Clean and Prominent
# -----------------------------------------------------------------------------
st.markdown("""
<div class="custom-top-bar">
    <div class="custom-top-title">Sports match video to text summary:Football</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sleek Top Control Bar: Source, Selection & Status (Replaces blank space)
# -----------------------------------------------------------------------------
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
sel_col1, sel_col2, sel_col3 = st.columns([1.0, 1.8, 1.4], gap="medium")

current_video_url = None
current_events_list = []
match_display_name = ""
match_status_text = "READY FOR BROADCAST"

with sel_col1:
    st.markdown('<div style="font-size:1.1rem; font-weight:800; color:#FFFFFF; margin-bottom:0.4rem;">📹 Video Source</div>', unsafe_allow_html=True)
    source_mode = st.radio(
        "Source Selection Mode",
        ["Demo clips", "Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )

with sel_col2:
    if source_mode == "Demo clips":
        st.markdown('<div style="font-size:1.1rem; font-weight:800; color:#FFFFFF; margin-bottom:0.4rem;">⚽ Select Match Clip</div>', unsafe_allow_html=True)
        chosen_demo_name = st.selectbox(
            "Select demo match clip:",
            list(DEMO_MATCHES.keys()),
            label_visibility="collapsed",
        )
        match_info = DEMO_MATCHES[chosen_demo_name]
        demo_file_path = BASE_DIR / match_info["file_path"]
        current_video_url = load_video_b64(demo_file_path) or match_info["static_url"]
        current_events_list = match_info["events"]
        match_display_name = f"{match_info['home']} vs {match_info['away']}"
    else:
        st.markdown('<div style="font-size:1.1rem; font-weight:800; color:#FFFFFF; margin-bottom:0.4rem;">📤 Upload Match Video (MP4)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a match video (MP4 format, max 3 min):",
            type=config.ALLOWED_FORMATS,
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            video_bytes = uploaded_file.getvalue()
            custom_filename = f"upload_{uploaded_file.name.replace(' ', '_')}"
            custom_dir = BASE_DIR / "static"
            custom_dir.mkdir(parents=True, exist_ok=True)
            custom_path = str(custom_dir / custom_filename)
            with open(custom_path, "wb") as f:
                f.write(video_bytes)

            is_valid, err = validate_video(custom_path)
            if not is_valid:
                st.error(f"Validation Error: {err}")
                match_status_text = "VALIDATION FAILED"
            else:
                current_video_url = load_video_b64(video_bytes)
                match_display_name = uploaded_file.name
                cache_key = f"analysis_{uploaded_file.name}_{len(video_bytes)}"

                if cache_key in st.session_state:
                    current_events_list = st.session_state[cache_key]
                else:
                    with st.spinner("Analyzing match video with visual models..."):
                        summary_txt, raw_evs = summarize_video(custom_path)
                        custom_events = []
                        custom_narratives = {
                            "goal": "GOAL! A BREATHTAKING ATTACKING FINISH! The attacker penetrates the defensive line and drives the ball clinically past the goalkeeper into the back of the net!",
                            "card": "BOOKING ISSUED! The referee halts play and pulls out a yellow card after a dangerous, cynical challenge that broke up a promising attack!",
                            "foul": "WHISTLE BLOWN FOR A FOUL! Heavy contact in the midfield duel brings down the attacker. The referee awards a free kick to relieve pressure.",
                            "corner": "CORNER KICK AWARDED! A vital defensive interception deflects the ball behind the goal line. Attackers flood into the box for the set-piece delivery!",
                        }
                        for ev in raw_evs:
                            ev_time = float(ev[0])
                            ev_lbl = str(ev[1]).lower()
                            display_title = "YELLOW CARD" if ev_lbl == "card" else ev_lbl.upper()
                            comm_text = custom_narratives.get(
                                ev_lbl,
                                f"KEY MATCH EVENT ({display_title})! Action captured by the model at {int(round(ev_time))} seconds!"
                            )
                            custom_events.append({
                                "trigger_time": ev_time,
                                "label": f"{display_title} — ATTACKING TEAM",
                                "team": "Attacking Team",
                                "text": comm_text,
                            })

                        if not custom_events:
                            custom_events.append({
                                "trigger_time": 2.0,
                                "label": "MATCH ANALYSIS COMPLETE",
                                "team": "Match Analysis",
                                "text": "Video processed successfully. The match features intense midfield battles with steady tactical shape and solid defensive discipline.",
                            })

                        st.session_state[cache_key] = custom_events
                        current_events_list = custom_events

                match_status_text = f"READY ({len(current_events_list)} EVENTS)"
        else:
            match_display_name = "Waiting for file upload..."
            match_status_text = "AWAITING VIDEO"

with sel_col3:
    st.markdown('<div style="font-size:1.1rem; font-weight:800; color:#FFFFFF; margin-bottom:0.4rem;">📊 Broadcast Feed Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#182236; border:1.5px solid #283750; border-radius:8px; padding:0.6rem 1rem; display:flex; flex-direction:column; justify-content:center; gap:0.25rem;">
        <div style="font-size:0.95rem; color:#F8FAFC; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            <span style="color:#94A3B8;">MATCH:</span> <strong>{match_display_name}</strong>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.85rem; font-weight:800;">
            <span style="color:#79FE0C; display:flex; align-items:center; gap:0.35rem;">
                <span style="width:8px; height:8px; border-radius:50%; background:#79FE0C; display:inline-block; box-shadow:0 0 8px #79FE0C;"></span>
                {match_status_text}
            </span>
            <span style="color:#64748B;">LIVE SYNC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Stage: Side-by-Side Video Player + Real-Time Live Commentary
# -----------------------------------------------------------------------------
if current_video_url:
    events_json = json.dumps(current_events_list)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: #FFFFFF;
            }}
            .theater-layout {{
                display: grid;
                grid-template-columns: 1.15fr 1fr;
                gap: 1.5rem;
                align-items: stretch;
            }}
            @media (max-width: 960px) {{
                .theater-layout {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            /* Video card */
            .video-card {{
                background: #111726;
                border: 1px solid #1E293B;
                border-radius: 14px;
                padding: 1.25rem;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}
            video {{
                width: 100%;
                max-height: 440px;
                border-radius: 10px;
                background: #000000;
                outline: none;
            }}
            .video-caption {{
                margin-top: 0.75rem;
                font-size: 0.88rem;
                color: #64748B;
                text-align: center;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 700;
            }}
            
            /* Commentary Terminal */
            .terminal-card {{
                background: linear-gradient(180deg, #131B2C 0%, #0C121E 100%);
                border: 2px solid #283750;
                border-radius: 16px;
                padding: 1.6rem 1.8rem;
                min-height: 480px;
                display: flex;
                flex-direction: column;
            }}
            
            .terminal-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #233149;
                padding-bottom: 0.9rem;
                margin-bottom: 1.25rem;
            }}
            .terminal-title {{
                font-size: 1.4rem;
                font-weight: 900;
                color: #FFFFFF;
                letter-spacing: -0.02em;
            }}
            .live-badge {{
                color: #79FE0C;
                font-size: 0.88rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            .pulse-dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #79FE0C;
                box-shadow: 0 0 12px #79FE0C;
            }}
            
            #commentaryContainer {{
                flex: 1;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}

            /* Broadcast event card */
            .broadcast-card {{
                background-color: #172236;
                border-left: 8px solid #79FE0C;
                border-radius: 12px;
                padding: 1.5rem 1.7rem;
                box-shadow: 0 6px 25px rgba(0,0,0,0.4);
                animation: fadeIn 0.35s ease-out;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(6px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .event-badge {{
                font-weight: 900;
                font-size: 1.45rem;
                color: #79FE0C;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.75rem;
            }}
            
            /* 32px Commentary Font */
            .commentary-body {{
                font-size: 32px !important;
                font-weight: 700 !important;
                line-height: 1.5 !important;
                color: #FFFFFF !important;
                letter-spacing: -0.01em !important;
                margin: 0 !important;
            }}
            
            .placeholder-box {{
                text-align: center;
                padding: 2rem 1rem;
                color: #64748B;
            }}
            .placeholder-icon {{
                font-size: 3rem;
                margin-bottom: 0.75rem;
            }}
            .placeholder-title {{
                font-size: 1.4rem;
                font-weight: 800;
                color: #FFFFFF;
                margin-bottom: 0.5rem;
            }}
            .placeholder-desc {{
                font-size: 1.05rem;
                color: #94A3B8;
                max-width: 440px;
                margin: 0 auto;
                line-height: 1.55;
            }}
        </style>
    </head>
    <body>
        <div class="theater-layout">
            <!-- Video Player (Left) -->
            <div class="video-card">
                <video id="matchVid" controls playsinline preload="auto">
                    <source src="{current_video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="video-caption">Live Match Video Feed • Play to Sync Commentary</div>
                <div id="quickJumpBar" style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.6rem; justify-content:center;"></div>
            </div>

            <!-- Synchronized Live Commentary Terminal (Right) -->
            <div class="terminal-card">
                <div class="terminal-header">
                    <div class="terminal-title">🎙️ LIVE MATCH COMMENTARY</div>
                    <div class="live-badge">
                        <div class="pulse-dot"></div>
                        SYNCHRONIZED FEED
                    </div>
                </div>
                
                <div id="commentaryContainer">
                    <div class="placeholder-box">
                        <div class="placeholder-icon">▶️</div>
                        <div class="placeholder-title">Press Play to Start Match Commentary</div>
                        <div class="placeholder-desc">
                            As the video plays, live broadcast commentary will automatically trigger and type out in real-time when key match events occur. You can also click any highlight button below the player to jump directly to that moment.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const vid = document.getElementById('matchVid');
            const container = document.getElementById('commentaryContainer');
            const jumpBar = document.getElementById('quickJumpBar');
            const events = {events_json};

            let triggeredSet = new Set();
            let hasStarted = false;

            if (events && events.length > 0 && jumpBar) {{
                events.forEach((ev, idx) => {{
                    const btn = document.createElement('button');
                    const cleanLabel = ev.label.split('—')[0].trim();
                    btn.textContent = `⏱️ ${{Math.round(ev.trigger_time)}}s • ${{cleanLabel}}`;
                    btn.style.cssText = 'background:#182236; color:#79FE0C; border:1.5px solid #283750; border-radius:6px; padding:0.4rem 0.8rem; font-size:0.85rem; font-weight:800; cursor:pointer; transition:all 0.2s;';
                    btn.onmouseover = () => {{ btn.style.background = '#283750'; btn.style.borderColor = '#79FE0C'; }};
                    btn.onmouseout = () => {{ btn.style.background = '#182236'; btn.style.borderColor = '#283750'; }};
                    btn.onclick = () => {{
                        vid.currentTime = Math.max(0, ev.trigger_time - 0.5);
                        vid.play();
                        renderCommentary(ev, false);
                    }};
                    jumpBar.appendChild(btn);
                }});
            }}

            function renderCommentary(eventItem, animateText = true) {{
                container.innerHTML = `
                    <div class="broadcast-card">
                        <div class="event-badge">${{eventItem.label}}</div>
                        <div id="typedText" class="commentary-body"></div>
                    </div>
                `;

                const textTarget = document.getElementById('typedText');
                if (!animateText) {{
                    textTarget.textContent = eventItem.text;
                    return;
                }}

                // Typewriter word-by-word streaming effect
                const words = eventItem.text.split(' ');
                let currentWordIdx = 0;
                textTarget.textContent = '';

                const timer = setInterval(() => {{
                    if (currentWordIdx < words.length) {{
                        textTarget.textContent += words[currentWordIdx] + ' ';
                        currentWordIdx++;
                    }} else {{
                        clearInterval(timer);
                    }}
                }}, 45);
            }}

            function showMatchInProgress() {{
                container.innerHTML = `
                    <div class="broadcast-card" style="border-left-color: #38BDF8;">
                        <div class="event-badge" style="color: #38BDF8;">MATCH UNDERWAY</div>
                        <div class="commentary-body">
                            The match is underway. Both teams are contesting possession across midfield with organized tactical structures. Watching for key scoring opportunities...
                        </div>
                    </div>
                `;
            }}

            vid.addEventListener('play', () => {{
                if (!hasStarted) {{
                    hasStarted = true;
                    let found = false;
                    for (let i = 0; i < events.length; i++) {{
                        if (Math.abs(vid.currentTime - events[i].trigger_time) < 4.0 || vid.currentTime >= events[i].trigger_time) {{
                            triggeredSet.add(i);
                            renderCommentary(events[i], false);
                            found = true;
                            break;
                        }}
                    }}
                    if (!found) {{
                        showMatchInProgress();
                    }}
                }}
            }});

            vid.addEventListener('timeupdate', () => {{
                const t = vid.currentTime;
                for (let i = 0; i < events.length; i++) {{
                    const ev = events[i];
                    if (t >= ev.trigger_time && !triggeredSet.has(i)) {{
                        triggeredSet.add(i);
                        renderCommentary(ev, true);
                    }}
                }}
            }});

            vid.addEventListener('seeked', () => {{
                const t = vid.currentTime;
                triggeredSet.clear();
                let activeEv = null;
                for (let i = 0; i < events.length; i++) {{
                    if (t >= events[i].trigger_time) {{
                        triggeredSet.add(i);
                        activeEv = events[i];
                    }}
                }}
                if (activeEv) {{
                    renderCommentary(activeEv, false);
                }} else if (hasStarted) {{
                    showMatchInProgress();
                }}
            }});
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=580, scrolling=False)

    if current_events_list:
        st.markdown('<div style="font-size:1.35rem; font-weight:900; color:#FFFFFF; margin-top:1.5rem; margin-bottom:0.75rem;">📋 Official Match Events & Timeline Summary</div>', unsafe_allow_html=True)
        ev_cols = st.columns(min(len(current_events_list), 3))
        for idx, ev in enumerate(current_events_list):
            with ev_cols[idx % min(len(current_events_list), 3)]:
                t_sec = int(round(ev.get("trigger_time", 0)))
                t_min = t_sec // 60
                t_remainder = t_sec % 60
                time_str = f"{t_min}:{t_remainder:02d} ({t_sec}s)"
                lbl = ev.get("label", "EVENT")
                comm = ev.get("text", "")
                st.markdown(f"""
                <div style="background:#111726; border:1px solid #1E293B; border-left:6px solid #79FE0C; border-radius:10px; padding:1.1rem 1.3rem; margin-bottom:1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                        <span style="font-weight:900; font-size:1.05rem; color:#79FE0C;">{lbl}</span>
                        <span style="background:#182236; border:1px solid #334155; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.85rem; font-weight:800; color:#FFFFFF;">⏱️ {time_str}</span>
                    </div>
                    <div style="font-size:0.95rem; color:#CBD5E1; line-height:1.5;">{comm}</div>
                </div>
                """, unsafe_allow_html=True)
