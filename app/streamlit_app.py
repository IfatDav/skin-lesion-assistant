"""Skin Lesion Triage · MVP for an investor demo.

Run from the repository root:
    streamlit run app/streamlit_app.py
(or with notebooks/10_mvp_demo.ipynb in Colab, which installs everything and gives a public address)

A patient sends a photo and six answers; the product ranks today's waiting list, the agent prepares the
case, and the dermatologist gets the morning list and a Word report. What is real and what is not:
app/engine.py, top of file.
"""
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

import engine as E

st.set_page_config(page_title="Skin Lesion Triage", layout="wide")
st.markdown("""<style>
.stApp, [data-testid="stSidebar"] {direction: rtl; text-align: right;}
.chip {display:inline-block;padding:3px 12px;border-radius:999px;font-size:13px;font-weight:600;margin-left:6px}
.ok {background:#DCEDE1;color:#2F7346} .warn {background:#F5E8CC;color:#8A5A00}
.big {font-size:30px;font-weight:800;margin:0}
.msg {background:rgba(12,104,120,.08);border-right:4px solid #0C6878;padding:10px 14px;border-radius:8px;font-size:17px}
</style>""", unsafe_allow_html=True)

EXAMPLES = Path(__file__).parent / "examples"
SAMPLES = {
    "סתירה בתשובות · בת 66": dict(file="contradiction.jpg", age=66, site="NECK", note="השומה החשיכה בחודש האחרון",
                                  ans=dict(itch="yes", grew="yes", hurt="no", changed="no", bleed="no", elevation="yes"), img=0.97),
    "נגע שמדמם · בן 72": dict(file="bleeding.jpg", age=72, site="NOSE", note="התחיל לדמם כשאני מתנגב במגבת",
                              ans=dict(itch="no", grew="yes", hurt="no", changed="yes", bleed="yes", elevation="yes"), img=0.93),
    "תמונה חשוכה · בן 64": dict(file="dark.jpg", age=64, site="FACE", note="",
                                ans=dict(itch="no", grew="yes", hurt="no", changed="yes", bleed="no", elevation="yes"), img=0.90),
    "שומה רגילה · בת 34": dict(file="nevus.jpg", age=34, site="ARM", note="",
                               ans=dict(itch="no", grew="no", hurt="no", changed="no", bleed="no", elevation="no"), img=0.15),
    "מטופל חדש (תמונה שלי)": None,
}
QUESTIONS = {"itch": "מגרד?", "grew": "גדל?", "hurt": "כואב?", "changed": "השתנה?", "bleed": "מדמם?",
             "elevation": "בולט מעל העור?"}
YES_NO = {"yes": "כן", "no": "לא", "unknown": "לא יודע/ת"}
DECISION_HE = {"proceed": "התיק מוכן לרופאה", "proceed_with_flag": "התיק מוכן לרופאה, עם הערה",
               "request_retake": "המטופל מתבקש לצלם שוב", "ask_patient": "המטופל מקבל שאלת הבהרה",
               "manual_review": "מועבר לבדיקה ידנית של רופא", "model_evidence_only": "סיכום קבוע (הסוכן לא זמין)"}
BAND_HE = {"very high": "גבוה מאוד", "high": "גבוה", "moderate": "בינוני", "low": "נמוך"}

live_ok, live_info = E.live_agent_ready()

# ── presenter options, out of the investor's way ───────────────────────────────────────────
with st.sidebar:
    with st.expander("אפשרויות למציג", expanded=False):
        agent_mode = st.radio("הסוכן", ["live", "rules", "off"], index=0 if live_ok else 1,
                              format_func={"live": "סוכן חי (Gemini)", "rules": "כללי ההחלטה, בלי API",
                                           "off": "כבוי: להראות שהמערכת עובדת גם בלעדיו"}.get)
        use_real_image_model = st.toggle("מודל התמונה האמיתי (ConvNeXt)", value=True)
        st.caption(f"סוכן חי: {'זמין' if live_ok else 'לא זמין'} ({live_info})")

# ── header ────────────────────────────────────────────────────────────────────────────────
st.title("Skin Lesion Triage")
st.markdown("**רשימת ההמתנה לרופא עור, מסודרת לפי סיכון.** מטופל מצלם נגע ועונה על שש שאלות; "
            "כל בוקר המערכת מעלה לראש התור את 10% הדחופים, והרופאה מחליטה.")
chips = [("סוכן AI: פעיל" if agent_mode == "live" and live_ok else "סוכן AI: מצב גיבוי", agent_mode == "live" and live_ok),
         ("מודל תמונה: ConvNeXt" if use_real_image_model else "מודל תמונה: ידני", use_real_image_model)]
st.markdown(" ".join(f"<span class='chip {'ok' if good else 'warn'}'>{t}</span>" for t, good in chips),
            unsafe_allow_html=True)

tab_patient, tab_inside, tab_doctor = st.tabs(["① המטופל שולח", "② מה המערכת עשתה", "③ מה הרופאה רואה"])

# ── 1. patient ────────────────────────────────────────────────────────────────────────────
with tab_patient:
    choice = st.selectbox("מטופל", list(SAMPLES), index=0)
    s = SAMPLES[choice]
    # Outside the form on purpose: a widget inside a form only reports its value on submit,
    # so an uploaded photo would not show (or replace the sample's) until "שליחה".
    upload = st.file_uploader("תמונה של הנגע", type=["jpg", "jpeg", "png"], key=f"up_{choice}",
                              help="תמונה שמעלים מחליפה את התמונה של המטופל לדוגמה")
    with st.form("send", border=True):
        left, right = st.columns([1, 1])
        with left:
            if upload is not None:
                st.image(Image.open(upload), caption="התמונה שהעלית: היא זו שתישלח", width=360)
            elif s:
                st.image(Image.open(EXAMPLES / s["file"]), caption="התמונה של המטופל לדוגמה", width=360)
            else:
                st.info("העלו תמונה של הנגע.")
        with right:
            age = st.number_input("גיל", 0, 110, value=(s or {}).get("age", 50), key=f"age_{choice}")
            site = st.selectbox("מיקום הנגע", list(E.SITES), format_func=E.SITES.get, key=f"site_{choice}",
                                index=list(E.SITES).index((s or {}).get("site", "BACK")))
            answers = {k: st.radio(q, list(YES_NO), horizontal=True, format_func=YES_NO.get, key=f"q_{k}_{choice}",
                                   index=list(YES_NO).index((s or {}).get("ans", {}).get(k, "no")))
                       for k, q in QUESTIONS.items()}
            note = st.text_area("רוצה להוסיף משהו?", value=(s or {}).get("note", ""), key=f"note_{choice}")
        sent = st.form_submit_button("שליחה", type="primary", use_container_width=True)

    if sent:
        src = upload or (EXAMPLES / s["file"] if s else None)
        if src is None:
            st.error("צריך תמונה: העלו תמונה של הנגע או בחרו מטופל לדוגמה.")
            st.stop()
        if upload is not None:
            upload.seek(0)                  # the preview above already read the file once
        photo = Image.open(src).convert("RGB")
        digest = hashlib.md5(photo.tobytes()).hexdigest()[:10]
        path = Path(tempfile.gettempdir()) / f"lesion_{digest}.jpg"
        photo.save(path)
        with st.status("המערכת עובדת…", expanded=True) as status:
            st.write("בודקת את איכות התמונה")
            quality = E.photo_quality(photo)
            st.write("מחשבת את ציון השאלון")
            q_score = E.questionnaire_score(age, site, {k: (v if v != "unknown" else None) for k, v in answers.items()})
            img_score, real_image = None, False
            if use_real_image_model:
                st.write("מריצה את מודל התמונה")
                img_score = E.image_score(str(path))
                real_image = img_score is not None
            if not real_image:
                img_score = (s or {}).get("img", 0.5)
            st.write("מדרגת את המטופל בתור של היום")
            r = E.rank_today(img_score, q_score)
            case = E.make_case("NEW-01", r["rank"], r, int(age), site, answers, note, str(path), quality)
            rv, how = None, "the agent runs only on the urgent list"
            if r["flagged"]:
                st.write("הסוכן מכין את התיק לרופאה")
                rv, how = E.review(case, agent_mode)
            status.update(label="נשלח. עברו ללשונית ② כדי לראות מה קרה.", state="complete", expanded=False)
        st.session_state["result"] = dict(photo=photo, quality=quality, q_score=q_score, img_score=img_score,
                                          real_image=real_image, r=r, case=case, rv=rv, how=how)
    st.caption("המטופל רואה רק: ״תודה, קיבלנו. רופא/ת עור יעברו על זה.״ הוא לא רואה ציון ולא אבחנה.")

res = st.session_state.get("result")

# ── 2. inside ─────────────────────────────────────────────────────────────────────────────
with tab_inside:
    if not res:
        st.info("שלחו מטופל בלשונית ①.")
    else:
        q, r, rv = res["quality"], res["r"], res["rv"]
        a, b = st.columns([1, 2])
        a.image(res["photo"], width=260)
        with b:
            c1, c2, c3 = st.columns(3)
            c1.metric("איכות התמונה", {"usable": "תקינה", "borderline": "גבולית", "unusable": "לא שמישה"}[q["status"]])
            c2.metric("ציון השאלון", f"{res['q_score']:.0%}")
            c3.metric("ציון התמונה", f"{res['img_score']:.2f}", help=None if res["real_image"] else "נקבע ידנית: מודל התמונה לא נטען")
            c1, c2, c3 = st.columns(3)
            c1.metric("מקום בתור היום", f"{r['rank']} מתוך {E.QUEUE_SIZE}")
            c2.metric("ציון משולב", f"{r['fused']:.2f}")
            c3.metric("רמת סיכון", BAND_HE[r["band"]])
        st.divider()
        if not r["flagged"]:
            st.markdown("<p class='big'>לא דחוף היום</p>", unsafe_allow_html=True)
            st.write("המטופל שומר על התור הרגיל, ומחר ידורג שוב. הסוכן מכין תיקים רק למטופלים ברשימה הדחופה.")
        else:
            st.markdown(f"<p class='big'>{DECISION_HE[rv.decision]}</p>", unsafe_allow_html=True)
            st.write(rv.decision_reason)
            msg = rv.follow_up_question or rv.retake_instruction
            if msg:
                st.markdown(f"<div class='msg'><b>הודעה למטופל:</b> {msg}</div>", unsafe_allow_html=True)
            with st.expander("מה הסוכן כתב לרופאה"):
                st.write(rv.evidence_summary)
                for c in rv.citations:
                    st.caption(f"“{c.quote}” · {c.source} · {c.id}")
                if rv.flags:
                    st.caption("בדיקות בקוד: " + "; ".join(rv.flags))
            st.caption(f"מנוע: {res['how']} · הציון והמקום בתור נעולים; הסוכן לא יכול לשנות אותם.")

# ── 3. doctor ─────────────────────────────────────────────────────────────────────────────
with tab_doctor:
    if not res:
        st.info("שלחו מטופל בלשונית ①.")
    else:
        r, rv = res["r"], res["rv"]
        st.subheader("רשימת הבוקר · 10 מתוך 100 פניות")
        top = r["queue"].head(E.SLOTS).copy()
        top["מטופל"] = top["patient_id"].replace({"NEW": "★ המטופל החדש"})
        top["רמת סיכון"] = top["band"].map(BAND_HE)
        st.dataframe(top[["rank", "מטופל", "רמת סיכון"]].rename(columns={"rank": "#"}), hide_index=True)
        if r["flagged"]:
            st.success("הרופאה תראה את המטופל מחר בבוקר. בסימולציית המרפאה: יום אחד במקום כ-24.")
            out = Path(tempfile.gettempdir()) / "triage_report.docx"
            E.word_report(res["case"], rv, str(out))
            st.download_button("הורדת דוח הבוקר (Word)", out.read_bytes(), file_name="triage_report.docx")
        else:
            st.info(f"המטופל במקום {r['rank']} ונשאר בתור הרגיל (כ-25 ימים בסימולציה).")
        st.caption("שאר המטופלים ברשימה: 99 מטופלי ולידציה אמיתיים מ-PAD-UFES-20, עם הציונים השמורים של המודלים.")
