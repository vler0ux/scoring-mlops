import gradio as gr
import os
from predict import load_model, predict
from logger import log_prediction

# ── Chargement du modèle au démarrage de l'API ────────────────────────────────
MODEL_URI = os.getenv("MODEL_URI", "../mlflow_model")
load_model(MODEL_URI)


def score_client(
    sk_id_curr,
    amt_income_total,
    amt_credit,
    amt_annuity,
    days_birth,
    days_employed,
    ext_source_1,
    ext_source_2,
    ext_source_3,
    code_gender,
    name_education_type,
):
    """Fonction appelée par Gradio à chaque prédiction."""

    input_data = {
        "SK_ID_CURR":           int(sk_id_curr),
        "AMT_INCOME_TOTAL":     float(amt_income_total),
        "AMT_CREDIT":           float(amt_credit),
        "AMT_ANNUITY":          float(amt_annuity),
        "DAYS_BIRTH":           float(days_birth),
        "DAYS_EMPLOYED":        float(days_employed),
        "EXT_SOURCE_1":         float(ext_source_1) if ext_source_1 else None,
        "EXT_SOURCE_2":         float(ext_source_2) if ext_source_2 else None,
        "EXT_SOURCE_3":         float(ext_source_3) if ext_source_3 else None,
        "CODE_GENDER":          code_gender,
        "NAME_EDUCATION_TYPE":  name_education_type,
    }

    try:
        result = predict(input_data)
        log_prediction(input_data, result)   # logging JSON pour le monitoring

        return (
            result["decision"],
            result["score"],
            result["risque_pct"],
            f"Seuil métier utilisé : {result['seuil']}"
        )

    except Exception as e:
        return f"Erreur : {str(e)}", None, None, None


# ── Interface Gradio ──────────────────────────────────────────────────────────
with gr.Blocks(title="Scoring Crédit — Prêt à Dépenser") as demo:

    gr.Markdown("## Scoring Crédit — Prêt à Dépenser")
    gr.Markdown("Renseignez les informations du client pour obtenir une décision de crédit.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Informations client")
            sk_id_curr          = gr.Number(label="ID Client (SK_ID_CURR)", value=100001)
            amt_income_total    = gr.Number(label="Revenu annuel (€)", value=135000)
            amt_credit          = gr.Number(label="Montant du crédit (€)", value=568800)
            amt_annuity         = gr.Number(label="Mensualité (€)", value=20250)
            days_birth          = gr.Number(label="Âge en jours (DAYS_BIRTH)", value=-12000)
            days_employed       = gr.Number(label="Ancienneté emploi en jours (DAYS_EMPLOYED)", value=-3000)
            ext_source_1        = gr.Number(label="Score externe 1 (EXT_SOURCE_1)", value=0.5)
            ext_source_2        = gr.Number(label="Score externe 2 (EXT_SOURCE_2)", value=0.6)
            ext_source_3        = gr.Number(label="Score externe 3 (EXT_SOURCE_3)", value=0.7)
            code_gender         = gr.Dropdown(["M", "F"], label="Genre", value="M")
            name_education_type = gr.Dropdown(
                ["Higher education", "Secondary / secondary special",
                 "Incomplete higher", "Lower secondary", "Academic degree"],
                label="Niveau d'éducation",
                value="Higher education"
            )
            btn = gr.Button("Calculer le score", variant="primary")

        with gr.Column():
            gr.Markdown("### Résultat")
            decision   = gr.Textbox(label="Décision")
            score      = gr.Number(label="Score de risque (probabilité de défaut)")
            risque_pct = gr.Number(label="Risque (%)")
            seuil_info = gr.Textbox(label="Info seuil")

    btn.click(
        fn=score_client,
        inputs=[
            sk_id_curr, amt_income_total, amt_credit, amt_annuity,
            days_birth, days_employed,
            ext_source_1, ext_source_2, ext_source_3,
            code_gender, name_education_type
        ],
        outputs=[decision, score, risque_pct, seuil_info]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)