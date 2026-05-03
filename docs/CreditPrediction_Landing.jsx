import { useState, useEffect, useRef } from "react";

const COLORS = {
  ink: "#0D1B2A",
  copper: "#C4622D",
  copperLight: "#D4784A",
  copperDark: "#A8521E",
  slateBlue: "#2E4057",
  sand: "#F0E6D3",
  sandDark: "#E0D0B5",
  sage: "#4A7C59",
  amber: "#D4A017",
  mist: "#F7F4F0",
  rule: "#D9D0C7",
  white: "#FFFFFF",
};

const useIntersection = (threshold = 0.15) => {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { threshold }
    );
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return [ref, visible];
};

const fadeUp = (visible, delay = 0) => ({
  opacity: visible ? 1 : 0,
  transform: visible ? "translateY(0)" : "translateY(28px)",
  transition: `opacity 0.7s ease ${delay}s, transform 0.7s ease ${delay}s`,
});

// ── Nav ──────────────────────────────────────────────────────────────
function Nav() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", fn);
    return () => window.removeEventListener("scroll", fn);
  }, []);

  return (
    <nav style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
      background: scrolled ? "rgba(13,27,42,0.97)" : "transparent",
      backdropFilter: scrolled ? "blur(12px)" : "none",
      borderBottom: scrolled ? `1px solid rgba(196,98,45,0.2)` : "none",
      transition: "all 0.4s ease",
      padding: "0 2rem",
    }}>
      <div style={{
        maxWidth: 1200, margin: "0 auto", display: "flex",
        alignItems: "center", justifyContent: "space-between",
        height: 68,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 6,
            background: `linear-gradient(135deg, ${COLORS.copper}, ${COLORS.slateBlue})`,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 12L8 4L13 12H3Z" fill="white" fillOpacity="0.9"/>
              <circle cx="8" cy="8" r="2" fill="white" fillOpacity="0.6"/>
            </svg>
          </div>
          <span style={{
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: 700, fontSize: 18, color: COLORS.white, letterSpacing: "-0.01em",
          }}>CreditLens</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
          {["Docs", "API", "GitHub"].map((item) => (
            <a key={item} href="#" style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 13, color: "rgba(255,255,255,0.65)",
              textDecoration: "none", letterSpacing: "0.02em",
              transition: "color 0.2s",
            }}
              onMouseEnter={e => e.target.style.color = COLORS.sand}
              onMouseLeave={e => e.target.style.color = "rgba(255,255,255,0.65)"}
            >{item}</a>
          ))}
          <a href="#" style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 13, color: COLORS.sand,
            border: `1px solid ${COLORS.copper}`,
            borderRadius: 4, padding: "8px 18px",
            textDecoration: "none", letterSpacing: "0.02em",
            transition: "all 0.2s",
          }}
            onMouseEnter={e => { e.target.style.background = COLORS.copper; e.target.style.color = COLORS.white; }}
            onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.color = COLORS.sand; }}
          >Try the Tool →</a>
        </div>
      </div>
    </nav>
  );
}

// ── Hero ─────────────────────────────────────────────────────────────
function Hero() {
  const [loaded, setLoaded] = useState(false);
  useEffect(() => { setTimeout(() => setLoaded(true), 100); }, []);

  return (
    <section style={{
      minHeight: "100vh",
      background: `radial-gradient(ellipse at 70% 50%, rgba(46,64,87,0.18) 0%, transparent 60%),
                   linear-gradient(160deg, ${COLORS.ink} 0%, #162336 60%, #1a2d45 100%)`,
      display: "flex", alignItems: "center",
      position: "relative", overflow: "hidden", paddingTop: 68,
    }}>
      {/* Background texture lines */}
      <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", opacity: 0.04 }}
        viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice">
        {Array.from({ length: 20 }).map((_, i) => (
          <line key={i} x1={i * 64} y1="0" x2={i * 64} y2="800" stroke="white" strokeWidth="0.5"/>
        ))}
        {Array.from({ length: 14 }).map((_, i) => (
          <line key={i} x1="0" y1={i * 60} x2="1200" y2={i * 60} stroke="white" strokeWidth="0.5"/>
        ))}
      </svg>

      {/* Copper accent orb */}
      <div style={{
        position: "absolute", right: "5%", top: "15%",
        width: 420, height: 420, borderRadius: "50%",
        background: `radial-gradient(circle, rgba(196,98,45,0.12) 0%, transparent 70%)`,
        filter: "blur(40px)",
      }} />

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 2rem", width: "100%" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 80, alignItems: "center" }}>

          {/* Left: Text */}
          <div>
            <div style={{ ...fadeUp(loaded, 0.1), marginBottom: 20 }}>
              <span style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 11, letterSpacing: "0.16em",
                color: COLORS.copper, textTransform: "uppercase",
                background: "rgba(196,98,45,0.1)",
                border: `1px solid rgba(196,98,45,0.3)`,
                padding: "5px 12px", borderRadius: 3,
              }}>ML-Powered Credit Intelligence</span>
            </div>

            <h1 style={{
              ...fadeUp(loaded, 0.2),
              fontFamily: "'Playfair Display', Georgia, serif",
              fontWeight: 700, fontSize: "clamp(42px, 5vw, 64px)",
              color: COLORS.white, lineHeight: 1.08,
              margin: "0 0 24px",
              letterSpacing: "-0.02em",
            }}>
              Know Before<br />
              <span style={{ color: COLORS.copper }}>You Lend.</span>
            </h1>

            <p style={{
              ...fadeUp(loaded, 0.3),
              fontFamily: "'Lora', Georgia, serif",
              fontSize: 18, color: "rgba(240,230,211,0.75)",
              lineHeight: 1.7, margin: "0 0 40px",
              maxWidth: 480,
            }}>
              Neural network–driven credit risk scoring and limit recommendation.
              Built for financial institutions that act on data, not guesswork.
            </p>

            <div style={{ ...fadeUp(loaded, 0.4), display: "flex", gap: 16, flexWrap: "wrap" }}>
              <a href="#" style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 14, letterSpacing: "0.03em",
                background: COLORS.copper,
                color: COLORS.white, padding: "14px 28px",
                borderRadius: 4, textDecoration: "none",
                border: `1px solid ${COLORS.copper}`,
                transition: "all 0.25s",
                display: "inline-block",
              }}
                onMouseEnter={e => e.target.style.background = COLORS.copperDark}
                onMouseLeave={e => e.target.style.background = COLORS.copper}
              >Try the Tool →</a>
              <a href="#" style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 14, letterSpacing: "0.03em",
                background: "transparent",
                color: COLORS.sand,
                padding: "14px 28px", borderRadius: 4,
                textDecoration: "none",
                border: `1px solid rgba(240,230,211,0.25)`,
                transition: "all 0.25s",
                display: "inline-block",
              }}
                onMouseEnter={e => e.target.style.borderColor = COLORS.sand}
                onMouseLeave={e => e.target.style.borderColor = "rgba(240,230,211,0.25)"}
              >View API Docs</a>
            </div>
          </div>

          {/* Right: Dashboard Preview / Illustration */}
          <div style={{ ...fadeUp(loaded, 0.35), position: "relative" }}>
            {/* Faux dashboard card */}
            <div style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 16, padding: 28, backdropFilter: "blur(4px)",
              position: "relative",
            }}>
              <div style={{
                position: "absolute", top: -1, left: 32, right: 32, height: 2,
                background: `linear-gradient(90deg, transparent, ${COLORS.copper}, transparent)`,
              }} />

              {/* Simulated analyst scene */}
              <div style={{
                width: "100%", aspectRatio: "16/10",
                borderRadius: 10, overflow: "hidden",
                background: `linear-gradient(135deg, #1a2e40 0%, #243b50 40%, #1d3345 100%)`,
                position: "relative", marginBottom: 20,
                border: "1px solid rgba(255,255,255,0.06)",
              }}>
                {/* Decorative analytical scene */}
                <svg viewBox="0 0 560 350" style={{ width: "100%", height: "100%" }}>
                  {/* Grid lines */}
                  {[70, 140, 210, 280, 350, 420, 490].map(x => (
                    <line key={x} x1={x} y1="0" x2={x} y2="350" stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
                  ))}
                  {[70, 140, 210, 280].map(y => (
                    <line key={y} x1="0" y1={y} x2="560" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
                  ))}

                  {/* Chart area — left panel */}
                  <rect x="20" y="20" width="240" height="200" rx="8" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5"/>
                  <text x="34" y="44" fontFamily="monospace" fontSize="9" fill="rgba(196,98,45,0.9)" letterSpacing="1">DEFAULT RISK</text>

                  {/* Semi-circle gauge */}
                  <g transform="translate(140, 160)">
                    <path d="M -70 0 A 70 70 0 0 1 70 0" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="10" strokeLinecap="round"/>
                    <path d="M -70 0 A 70 70 0 0 1 -28 -60" fill="none" stroke={COLORS.sage} strokeWidth="10" strokeLinecap="round"/>
                    <text x="0" y="-18" textAnchor="middle" fontFamily="monospace" fontSize="22" fill="white" fontWeight="600">15%</text>
                    <text x="0" y="-2" textAnchor="middle" fontFamily="monospace" fontSize="9" fill={COLORS.sage} letterSpacing="1">LOW RISK</text>
                    {/* Needle */}
                    <line x1="0" y1="0" x2="-38" y2="-57" stroke={COLORS.copper} strokeWidth="2" strokeLinecap="round"/>
                    <circle cx="0" cy="0" r="5" fill={COLORS.copper}/>
                  </g>

                  {/* Right panel */}
                  <rect x="276" y="20" width="264" height="200" rx="8" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5"/>
                  <text x="292" y="44" fontFamily="monospace" fontSize="9" fill="rgba(196,98,45,0.9)" letterSpacing="1">CREDIT LIMIT</text>

                  {/* Bar chart */}
                  {[
                    { label: "Current", val: 100, color: "rgba(46,64,87,0.8)", y: 80 },
                    { label: "Recommended", val: 150, color: COLORS.copper, y: 120 },
                  ].map(({ label, val, color, y }) => (
                    <g key={label}>
                      <text x="292" y={y - 4} fontFamily="monospace" fontSize="8" fill="rgba(255,255,255,0.4)">{label}</text>
                      <rect x="292" y={y} width={val * 1.5} height="16" rx="3" fill={color}/>
                      <text x={292 + val * 1.5 + 6} y={y + 12} fontFamily="monospace" fontSize="9" fill="rgba(255,255,255,0.7)">{val}K</text>
                    </g>
                  ))}

                  <text x="292" y="165" fontFamily="monospace" fontSize="8" fill="rgba(255,255,255,0.35)">Adjustment Factor</text>
                  <text x="292" y="180" fontFamily="monospace" fontSize="18" fill={COLORS.copper} fontWeight="700">×1.5</text>

                  {/* Bottom factors strip */}
                  <rect x="20" y="236" width="520" height="94" rx="8" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.06)" strokeWidth="0.5"/>
                  <text x="34" y="258" fontFamily="monospace" fontSize="8" fill="rgba(196,98,45,0.7)" letterSpacing="1">RECOMMENDATION FACTORS</text>

                  {[
                    { label: "Good payment history", x: 34 },
                    { label: "High pay / bill ratio", x: 190 },
                    { label: "Education positive", x: 346 },
                  ].map(({ label, x }) => (
                    <g key={label}>
                      <circle cx={x + 5} cy="276" r="3.5" fill={COLORS.sage}/>
                      <text x={x + 14} y="279" fontFamily="monospace" fontSize="8.5" fill="rgba(255,255,255,0.65)">{label}</text>
                    </g>
                  ))}

                  {/* Analyst silhouette / abstract figure */}
                  <g transform="translate(470, 60)" opacity="0.15">
                    <circle cx="0" cy="0" r="18" fill="white"/>
                    <rect x="-22" y="22" width="44" height="50" rx="8" fill="white"/>
                  </g>
                </svg>
              </div>

              {/* Cards row */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {[
                  { label: "Default Risk", value: "15.00%", tag: "Low Risk", tagColor: COLORS.sage },
                  { label: "Credit Limit", value: "₦150,000", tag: "+50% increase", tagColor: COLORS.copper },
                ].map(({ label, value, tag, tagColor }) => (
                  <div key={label} style={{
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 8, padding: "14px 16px",
                  }}>
                    <div style={{
                      fontFamily: "'IBM Plex Mono', monospace",
                      fontSize: 10, color: "rgba(240,230,211,0.45)",
                      letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6,
                    }}>{label}</div>
                    <div style={{
                      fontFamily: "'IBM Plex Mono', monospace",
                      fontSize: 22, fontWeight: 600, color: COLORS.white, marginBottom: 6,
                    }}>{value}</div>
                    <span style={{
                      fontFamily: "'IBM Plex Mono', monospace",
                      fontSize: 10, color: tagColor,
                      background: `${tagColor}18`,
                      border: `1px solid ${tagColor}40`,
                      padding: "2px 8px", borderRadius: 3,
                    }}>{tag}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats strip */}
      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0,
        background: "rgba(0,0,0,0.25)",
        borderTop: "1px solid rgba(255,255,255,0.07)",
        padding: "18px 2rem",
      }}>
        <div style={{
          maxWidth: 1200, margin: "0 auto",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 60,
        }}>
          {[
            { val: "98.2%", label: "Model Accuracy" },
            { val: "<200ms", label: "API Latency" },
            { val: "MIT", label: "Open Source" },
            { val: "TF/Keras", label: "Neural Networks" },
          ].map(({ val, label }) => (
            <div key={label} style={{ textAlign: "center" }}>
              <div style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 20, fontWeight: 600, color: COLORS.copper,
              }}>{val}</div>
              <div style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 10, color: "rgba(255,255,255,0.4)",
                letterSpacing: "0.1em", textTransform: "uppercase", marginTop: 2,
              }}>{label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── How It Works ─────────────────────────────────────────────────────
function HowItWorks() {
  const [ref, visible] = useIntersection();
  const steps = [
    { n: "01", title: "Submit Customer Data", body: "Pass credit limit, age, payment history, bill and payment amounts via a clean REST endpoint or our dashboard form." },
    { n: "02", title: "ML Engine Scores Risk", body: "Deep neural networks with dropout regularization and random oversampling assess default probability and optimal credit limits." },
    { n: "03", title: "Receive Actionable Insight", body: "Get a risk percentage, risk tier classification, and explicit recommendation factors — ready for your underwriting workflow." },
  ];

  return (
    <section style={{ background: COLORS.mist, padding: "120px 2rem", borderTop: `1px solid ${COLORS.rule}` }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }} ref={ref}>
        <div style={{ ...fadeUp(visible), textAlign: "center", marginBottom: 72 }}>
          <span style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 11, letterSpacing: "0.16em", color: COLORS.copper,
            textTransform: "uppercase",
          }}>How It Works</span>
          <h2 style={{
            fontFamily: "'DM Serif Display', 'Playfair Display', Georgia, serif",
            fontSize: "clamp(30px, 4vw, 44px)", color: COLORS.ink,
            margin: "12px 0 0", fontWeight: 400,
          }}>From data to decision in three steps.</h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 2, position: "relative" }}>
          {/* Connector line */}
          <div style={{
            position: "absolute", top: 36, left: "16.5%", right: "16.5%", height: 1,
            background: `linear-gradient(90deg, ${COLORS.rule}, ${COLORS.copper}80, ${COLORS.rule})`,
          }} />

          {steps.map(({ n, title, body }, i) => (
            <div key={n} style={{
              ...fadeUp(visible, 0.1 + i * 0.12),
              padding: "0 32px", textAlign: "center",
            }}>
              <div style={{
                width: 72, height: 72, borderRadius: "50%",
                background: i === 1 ? COLORS.copper : COLORS.white,
                border: `2px solid ${i === 1 ? COLORS.copper : COLORS.rule}`,
                display: "flex", alignItems: "center", justifyContent: "center",
                margin: "0 auto 28px",
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 15, fontWeight: 600,
                color: i === 1 ? COLORS.white : COLORS.copper,
                position: "relative", zIndex: 1,
              }}>{n}</div>
              <h3 style={{
                fontFamily: "'DM Serif Display', 'Playfair Display', Georgia, serif",
                fontSize: 20, color: COLORS.ink, fontWeight: 400,
                margin: "0 0 12px",
              }}>{title}</h3>
              <p style={{
                fontFamily: "'Lora', Georgia, serif",
                fontSize: 15, color: "#5A6878", lineHeight: 1.7, margin: 0,
              }}>{body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── Features ─────────────────────────────────────────────────────────
function Features() {
  const [ref, visible] = useIntersection();
  const features = [
    {
      icon: "⬡",
      title: "Default Prediction",
      desc: "Binary classification with probability output. Handles imbalanced datasets via random oversampling — giving reliable scores even for rare default events.",
      tags: ["Binary Classification", "Probability Score", "Oversampling"],
      accent: COLORS.copper,
    },
    {
      icon: "◈",
      title: "Credit Limit Recommendation",
      desc: "Regression-based engine estimates appropriate limits from payment history, age, education, and bill ratios — with full adjustment factor breakdown.",
      tags: ["Regression Model", "Custom Feature Eng.", "Regularization"],
      accent: COLORS.slateBlue,
    },
    {
      icon: "◎",
      title: "FastAPI Backend",
      desc: "Asynchronous REST API with Pydantic validation and robust error handling. Integrates into your existing workflow in minutes with clean JSON responses.",
      tags: ["FastAPI", "Pydantic", "<200ms Latency"],
      accent: COLORS.sage,
    },
    {
      icon: "◇",
      title: "Transparent Risk Factors",
      desc: "Every prediction ships with human-readable recommendation factors — not just a number, but the reasoning behind it for your compliance team.",
      tags: ["Explainability", "Risk Factors", "Audit Trail"],
      accent: COLORS.amber,
    },
  ];

  return (
    <section style={{ background: COLORS.white, padding: "120px 2rem", borderTop: `1px solid ${COLORS.rule}` }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }} ref={ref}>
        <div style={{ ...fadeUp(visible), marginBottom: 72 }}>
          <span style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 11, letterSpacing: "0.16em", color: COLORS.copper,
            textTransform: "uppercase",
          }}>Capabilities</span>
          <h2 style={{
            fontFamily: "'DM Serif Display', 'Playfair Display', Georgia, serif",
            fontSize: "clamp(30px, 4vw, 44px)", color: COLORS.ink,
            margin: "12px 0 0", fontWeight: 400, maxWidth: 540,
          }}>Everything your credit team needs to move with confidence.</h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 24 }}>
          {features.map(({ icon, title, desc, tags, accent }, i) => (
            <div key={title} style={{
              ...fadeUp(visible, 0.1 + i * 0.1),
              background: COLORS.mist,
              border: `1px solid ${COLORS.rule}`,
              borderLeft: `3px solid ${accent}`,
              borderRadius: 12, padding: 32,
              transition: "box-shadow 0.25s, transform 0.25s",
              cursor: "default",
            }}
              onMouseEnter={e => {
                e.currentTarget.style.boxShadow = `0 8px 32px rgba(13,27,42,0.1)`;
                e.currentTarget.style.transform = "translateY(-2px)";
              }}
              onMouseLeave={e => {
                e.currentTarget.style.boxShadow = "none";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              <div style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 28, color: accent, marginBottom: 16,
              }}>{icon}</div>
              <h3 style={{
                fontFamily: "'DM Serif Display', 'Playfair Display', Georgia, serif",
                fontSize: 22, color: COLORS.ink, fontWeight: 400,
                margin: "0 0 12px",
              }}>{title}</h3>
              <p style={{
                fontFamily: "'Lora', Georgia, serif",
                fontSize: 15, color: "#5A6878", lineHeight: 1.7,
                margin: "0 0 20px",
              }}>{desc}</p>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {tags.map(tag => (
                  <span key={tag} style={{
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontSize: 10, color: accent,
                    background: `${accent}12`,
                    border: `1px solid ${accent}30`,
                    padding: "3px 10px", borderRadius: 3,
                    letterSpacing: "0.04em",
                  }}>{tag}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── API Preview ───────────────────────────────────────────────────────
function ApiPreview() {
  const [ref, visible] = useIntersection();
  const [active, setActive] = useState(0);

  const snippets = [
    {
      label: "Default Risk",
      code: `curl -X POST http://localhost:8000/api/v1/predict/default \\
  -H "Content-Type: application/json" \\
  -d '{
    "credit_limit": 100000,
    "age": 25,
    "payment_status": 0,
    "bill_amount": 10000,
    "payment_amount": 2000
  }'`,
      response: `{
  "prediction": "Low Risk of Default",
  "probability": 0.15,
  "is_high_risk": false,
  "risk_factors": [
    "Low payment to bill ratio"
  ]
}`,
    },
    {
      label: "Credit Limit",
      code: `curl -X POST http://localhost:8000/api/v1/predict/credit-limit \\
  -H "Content-Type: application/json" \\
  -d '{
    "credit_limit": 100000,
    "age": 35,
    "education": 1,
    "payment_amount": 8000
  }'`,
      response: `{
  "predicted_credit_limit": 150000,
  "adjustment_factor": 1.5,
  "recommendation_factors": [
    "Good payment history",
    "High payment to bill ratio"
  ]
}`,
    },
  ];

  return (
    <section style={{
      background: COLORS.ink, padding: "120px 2rem",
      borderTop: `1px solid rgba(196,98,45,0.2)`,
    }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }} ref={ref}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 80, alignItems: "start" }}>
          <div>
            <div style={{ ...fadeUp(visible), marginBottom: 24 }}>
              <span style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 11, letterSpacing: "0.16em", color: COLORS.copper,
                textTransform: "uppercase",
              }}>API Reference</span>
              <h2 style={{
                fontFamily: "'DM Serif Display', 'Playfair Display', Georgia, serif",
                fontSize: "clamp(28px, 3.5vw, 42px)", color: COLORS.white,
                margin: "12px 0 20px", fontWeight: 400, lineHeight: 1.2,
              }}>Simple endpoints. Powerful predictions.</h2>
              <p style={{
                fontFamily: "'Lora', Georgia, serif",
                fontSize: 16, color: "rgba(240,230,211,0.6)", lineHeight: 1.7,
              }}>Three endpoints. Clean JSON in and out. Integrate with any language or tool in minutes.</p>
            </div>

            {[
              { method: "GET", path: "/api/v1/health", desc: "Service health & model status" },
              { method: "POST", path: "/api/v1/predict/default", desc: "Default risk probability" },
              { method: "POST", path: "/api/v1/predict/credit-limit", desc: "Credit limit recommendation" },
            ].map(({ method, path, desc }, i) => (
              <div key={path} style={{
                ...fadeUp(visible, 0.1 + i * 0.08),
                display: "flex", alignItems: "flex-start", gap: 16,
                padding: "16px 0",
                borderBottom: `1px solid rgba(255,255,255,0.06)`,
              }}>
                <span style={{
                  fontFamily: "'IBM Plex Mono', monospace",
                  fontSize: 10, fontWeight: 600,
                  color: method === "GET" ? COLORS.sage : COLORS.copper,
                  background: method === "GET" ? `${COLORS.sage}18` : `${COLORS.copper}18`,
                  border: `1px solid ${method === "GET" ? COLORS.sage : COLORS.copper}40`,
                  padding: "3px 10px", borderRadius: 3, letterSpacing: "0.08em",
                  flexShrink: 0, marginTop: 2,
                }}>{method}</span>
                <div>
                  <div style={{
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontSize: 13, color: "rgba(255,255,255,0.85)", marginBottom: 3,
                  }}>{path}</div>
                  <div style={{
                    fontFamily: "'Lora', Georgia, serif",
                    fontSize: 13, color: "rgba(255,255,255,0.4)",
                  }}>{desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Code panel */}
          <div style={{ ...fadeUp(visible, 0.25) }}>
            <div style={{
              background: "#0A1520",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 12, overflow: "hidden",
            }}>
              {/* Tab bar */}
              <div style={{
                display: "flex", alignItems: "center",
                background: "rgba(0,0,0,0.3)",
                borderBottom: "1px solid rgba(255,255,255,0.06)",
                padding: "0 16px",
              }}>
                <div style={{ display: "flex", gap: 6, padding: "14px 0", marginRight: 20 }}>
                  {["#F7685B", "#FDBC40", "#34C84A"].map(c => (
                    <div key={c} style={{ width: 10, height: 10, borderRadius: "50%", background: c }}/>
                  ))}
                </div>
                {snippets.map(({ label }, i) => (
                  <button key={label} onClick={() => setActive(i)} style={{
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontSize: 11, letterSpacing: "0.06em",
                    background: "none", border: "none", cursor: "pointer",
                    color: i === active ? COLORS.copper : "rgba(255,255,255,0.35)",
                    padding: "14px 14px",
                    borderBottom: i === active ? `2px solid ${COLORS.copper}` : "2px solid transparent",
                    transition: "all 0.2s",
                  }}>{label}</button>
                ))}
              </div>

              <div style={{ padding: 24 }}>
                <div style={{
                  fontFamily: "'IBM Plex Mono', monospace",
                  fontSize: 12, lineHeight: 1.7,
                  color: "rgba(240,230,211,0.75)",
                  whiteSpace: "pre", overflowX: "auto",
                }}>{snippets[active].code}</div>

                <div style={{
                  margin: "20px 0 0",
                  borderTop: "1px solid rgba(255,255,255,0.06)",
                  paddingTop: 20,
                }}>
                  <div style={{
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontSize: 10, color: COLORS.sage, letterSpacing: "0.1em",
                    textTransform: "uppercase", marginBottom: 12,
                  }}>Response</div>
                  <div style={{
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontSize: 12, lineHeight: 1.7,
                    color: "rgba(240,230,211,0.6)",
                    whiteSpace: "pre", overflowX: "auto",
                  }}>{snippets[active].response}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ── CTA ───────────────────────────────────────────────────────────────
function CTA() {
  const [ref, visible] = useIntersection();
  return (
    <section style={{
      background: `linear-gradient(135deg, ${COLORS.copper} 0%, ${COLORS.copperDark} 100%)`,
      padding: "100px 2rem",
      textAlign: "center",
      position: "relative", overflow: "hidden",
    }}>
      {/* Texture */}
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: `repeating-linear-gradient(45deg, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 20px)`,
      }}/>
      <div ref={ref} style={{ position: "relative" }}>
        <h2 style={{
          ...fadeUp(visible),
          fontFamily: "'Playfair Display', Georgia, serif",
          fontSize: "clamp(32px, 4vw, 52px)", color: COLORS.white,
          fontWeight: 700, margin: "0 0 20px", letterSpacing: "-0.02em",
        }}>Start predicting smarter credit decisions.</h2>
        <p style={{
          ...fadeUp(visible, 0.1),
          fontFamily: "'Lora', Georgia, serif",
          fontSize: 18, color: "rgba(255,255,255,0.8)",
          margin: "0 auto 40px", maxWidth: 480, lineHeight: 1.6,
        }}>Open source, MIT licensed, and ready to deploy. Clone the repo and run your first prediction in under five minutes.</p>
        <div style={{ ...fadeUp(visible, 0.2), display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
          <a href="https://github.com/austinLorenzMccoy/credit-default-prediction" target="_blank" rel="noreferrer" style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 14, letterSpacing: "0.03em",
            background: COLORS.white, color: COLORS.copper,
            padding: "14px 32px", borderRadius: 4,
            textDecoration: "none", fontWeight: 600,
            transition: "opacity 0.2s",
          }}
            onMouseEnter={e => e.target.style.opacity = "0.9"}
            onMouseLeave={e => e.target.style.opacity = "1"}
          >View on GitHub →</a>
          <a href="#" style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 14, letterSpacing: "0.03em",
            background: "transparent", color: COLORS.white,
            border: "1px solid rgba(255,255,255,0.4)",
            padding: "14px 32px", borderRadius: 4,
            textDecoration: "none",
            transition: "border-color 0.2s",
          }}
            onMouseEnter={e => e.target.style.borderColor = "white"}
            onMouseLeave={e => e.target.style.borderColor = "rgba(255,255,255,0.4)"}
          >Read the Docs</a>
        </div>
      </div>
    </section>
  );
}

// ── Footer ────────────────────────────────────────────────────────────
function Footer() {
  return (
    <footer style={{
      background: COLORS.ink, padding: "48px 2rem",
      borderTop: "1px solid rgba(255,255,255,0.06)",
    }}>
      <div style={{
        maxWidth: 1200, margin: "0 auto",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexWrap: "wrap", gap: 24,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 5,
            background: `linear-gradient(135deg, ${COLORS.copper}, ${COLORS.slateBlue})`,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
              <path d="M3 12L8 4L13 12H3Z" fill="white" fillOpacity="0.9"/>
            </svg>
          </div>
          <span style={{
            fontFamily: "'Playfair Display', Georgia, serif",
            fontWeight: 700, fontSize: 16, color: COLORS.white,
          }}>CreditLens</span>
          <span style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 10, color: "rgba(255,255,255,0.3)", letterSpacing: "0.08em",
          }}>MIT License</span>
        </div>

        <div style={{ display: "flex", gap: 28 }}>
          {[
            { label: "GitHub", href: "https://github.com/austinLorenzMccoy/credit-default-prediction" },
            { label: "chibuezeaugustine23@gmail.com", href: "mailto:chibuezeaugustine23@gmail.com" },
            { label: "API Docs", href: "#" },
          ].map(({ label, href }) => (
            <a key={label} href={href} style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 12, color: "rgba(255,255,255,0.4)",
              textDecoration: "none", letterSpacing: "0.04em",
              transition: "color 0.2s",
            }}
              onMouseEnter={e => e.target.style.color = COLORS.sand}
              onMouseLeave={e => e.target.style.color = "rgba(255,255,255,0.4)"}
            >{label}</a>
          ))}
        </div>
      </div>
    </footer>
  );
}

// ── App ───────────────────────────────────────────────────────────────
export default function App() {
  useEffect(() => {
    // Load Google Fonts
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Serif+Display&family=IBM+Plex+Mono:wght@400;500;600&family=Lora:wght@400;500&display=swap";
    document.head.appendChild(link);

    document.title = "CreditLens — ML Credit Intelligence";
    document.body.style.margin = "0";
    document.body.style.padding = "0";
    document.body.style.background = COLORS.ink;
  }, []);

  return (
    <div style={{ fontFamily: "sans-serif" }}>
      <Nav />
      <Hero />
      <HowItWorks />
      <Features />
      <ApiPreview />
      <CTA />
      <Footer />
    </div>
  );
}
