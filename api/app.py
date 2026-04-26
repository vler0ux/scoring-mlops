import gradio as gr
import os
from predict import load_model, predict, OPTIMAL_THRESHOLD
from logger import log_prediction

MODEL_URI = os.getenv("MODEL_URI", "../mlflow_model")
load_model(MODEL_URI)


def build_gauge_html(score: float, seuil: float) -> str:
    """Génère une jauge HTML visuelle score vs seuil."""
    pct = round(score * 100, 1)
    seuil_pct = round(seuil * 100, 1)
    color = "#e74c3c" if score >= seuil else "#27ae60"

    return f"""
    <div style="font-family:sans-serif; padding:16px; background:#f9f9f9;
                border-radius:10px; border:1px solid #ddd;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-weight:bold; color:{color}; font-size:1.1em;">
                Score de risque : {pct}%
            </span>
            <span style="color:#555; font-size:0.95em;">
                Seuil métier : {seuil_pct}%
            </span>
        </div>

        <!-- Barre de progression -->
        <div style="position:relative; height:28px; background:#e0e0e0;
                    border-radius:14px; overflow:visible;">
            <!-- Remplissage score -->
            <div style="width:{pct}%; height:100%; background:{color};
                        border-radius:14px; transition:width 0.4s;"></div>
            <!-- Marqueur seuil -->
            <div style="position:absolute; top:-6px; left:{seuil_pct}%;
                        transform:translateX(-50%);
                        width:3px; height:40px; background:#2c3e50;
                        border-radius:2px;">
            </div>
            <div style="position:absolute; top:32px; left:{seuil_pct}%;
                        transform:translateX(-50%);
                        font-size:0.78em; color:#2c3e50; white-space:nowrap; font-weight:bold;">
                ▲ Seuil {seuil_pct}%
            </div>
        </div>

        <!-- Décision -->
        <div style="margin-top:36px; font-size:1.3em; font-weight:bold;
                    color:{color}; text-align:center;">
            {"❌ Crédit REFUSÉ" if score >= seuil else "✅ Crédit ACCORDÉ"}
        </div>

        <!-- Légende -->
        <div style="margin-top:8px; display:flex; justify-content:space-between;
                    font-size:0.78em; color:#888;">
            <span>0% — Risque nul</span>
            <span>100% — Défaut certain</span>
        </div>
    </div>
    """


def score_client(
    amt_income_total,
    amt_credit,
    amt_annuity,
    age_ans,           # ← en années maintenant
    anciennete_ans,    # ← en années maintenant
    sans_emploi,       # ← checkbox "Sans emploi / Retraité"
    ext_source_1,
    ext_source_2,
    ext_source_3,
    code_gender,
    name_education_type,
):
    # Conversion années → jours pour le modèle
    days_birth = int(age_ans * 365.25)

    if sans_emploi:
        days_employed = 365243   # valeur spéciale Home Credit = sans emploi
    else:
        days_employed = int(anciennete_ans * 365.25)

    input_data = {
        "AMT_INCOME_TOTAL":    float(amt_income_total),
        "AMT_CREDIT":          float(amt_credit),
        "AMT_ANNUITY":         float(amt_annuity),
        "DAYS_BIRTH":          float(days_birth),
        "DAYS_EMPLOYED":       float(days_employed),
        "EXT_SOURCE_1": float(ext_source_1) if ext_source_1 is not None else None,
        "EXT_SOURCE_2": float(ext_source_2) if ext_source_2 is not None else None,
        "EXT_SOURCE_3": float(ext_source_3) if ext_source_3 is not None else None,
        "CODE_GENDER":         code_gender,
        "NAME_EDUCATION_TYPE": name_education_type,
    }

    try:
        import time
        t0 = time.time()
        result = predict(input_data)
        inference_time_ms = round((time.time() - t0) * 1000, 2)
        #print(f"DEBUG → proba brute = {result['score']}, type = {type(result['score'])}")
        #print(f"DEBUG → score={result['score']}, seuil={result['seuil']}, décision={result['decision']}")
        
        log_prediction(input_data, result)
        gauge = build_gauge_html(result["score"], result["seuil"])
        return gauge, result["score"], result["risque_pct"]

    except Exception as e:
        return f"<p style='color:red'>Erreur : {str(e)}</p>", None, None


# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Scoring Crédit — Prêt à Dépenser") as demo:

    gr.Markdown("## 🏦 Scoring Crédit — Prêt à Dépenser")
    gr.Markdown("Renseignez les informations du client pour obtenir une décision de crédit.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Informations client")
            amt_income_total = gr.Number(label="Revenu annuel (€)", value=135000)
            amt_credit       = gr.Number(label="Montant du crédit (€)", value=200000)
            amt_annuity      = gr.Number(label="Mensualité (€)", value=8000)

            # ── Âge et ancienneté en ANNÉES ──────────────────────────────
            age_ans = gr.Slider(
                label="Âge du client (années)",
                minimum=18, maximum=75, step=1, value=35
            )
            anciennete_ans = gr.Slider(
                label="Ancienneté dans l'emploi actuel (années)",
                minimum=0, maximum=40, step=1, value=8
            )
            sans_emploi = gr.Checkbox(
                label="Sans emploi / Retraité / Étudiant",
                value=False
            )

            gr.Markdown("### Scores externes")
            ext_source_1 = gr.Slider(label="EXT_SOURCE_1", minimum=0, maximum=1, step=0.01, value=0.5)
            ext_source_2 = gr.Slider(label="EXT_SOURCE_2", minimum=0, maximum=1, step=0.01, value=0.6)
            ext_source_3 = gr.Slider(label="EXT_SOURCE_3", minimum=0, maximum=1, step=0.01, value=0.7)

            gr.Markdown("### Profil")
            code_gender = gr.Dropdown(["M", "F"], label="Genre", value="M")
            name_education_type = gr.Dropdown(
                ["Higher education", "Secondary / secondary special",
                "Incomplete higher", "Lower secondary", "Academic degree"],
                label="Niveau d'éducation", value="Higher education"
            )
            btn = gr.Button("Calculer le score", variant="primary")

        with gr.Column():
            gr.Markdown("### Résultat")
            jauge    = gr.HTML(label="Score vs Seuil métier")
            score    = gr.Number(label="Score brut (0–1)")
            risque   = gr.Number(label="Risque (%)")

    btn.click(
        fn=score_client,
        inputs=[
            amt_income_total, amt_credit, amt_annuity,
            age_ans, anciennete_ans, sans_emploi,
            ext_source_1, ext_source_2, ext_source_3,
            code_gender, name_education_type
        ],
        outputs=[jauge, score, risque]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)