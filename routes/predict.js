import { Router } from "express";

const router = Router();
const ML_URL = process.env.ML_URL || "http://localhost:8000";

router.post("/", async (req, res) => {
  const subject = (req.body.subject || "").trim();
  const content = (req.body.content || "").trim();

  if (!content) {
    return res.status(400).render("home", {
      title: "Home",
      error: "Paste the email content to analyze it.",
      subject,
      content,
    });
  }

  try {
    const response = await fetch(`${ML_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, content }),
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error(`Model service returned ${response.status}`);
    const result = await response.json();

    return res.render("result", {
      title: "Result",
      subject,
      content,
      isPhishing: result.label === "phishing",
      phishingPct: Math.round(result.probability * 100),
      warningSigns: result.warning_signs,
    });
  } catch (e) {
    console.error("Prediction failed:", e.message);
    return res.status(503).render("home", {
      title: "Home",
      error:
        "The model service isn't running. Start it with: cd ml && uvicorn serve:app --port 8000",
      subject,
      content,
    });
  }
});

export default router;
