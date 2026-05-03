# 📋 Frontend PRD — Credit Prediction ML API
**Product Requirements Document · v1.0**
**Project:** Credit Prediction Machine Learning API
**Author:** Product Design Team
**Date:** 2026-05-03

---

## 1. Executive Summary

This document defines the frontend product requirements and wireframe specifications for the Credit Prediction ML API. The interface surfaces two core ML capabilities — **credit default risk scoring** and **credit limit recommendation** — in a clean, trust-inspiring dashboard designed for financial analysts, underwriters, and credit officers.

The design philosophy: **data-driven confidence**. Every screen reduces cognitive load, surfaces actionable insights, and communicates risk with clarity.

---

## 2. Design System

### 2.1 Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `--ink` | `#0D1B2A` | Primary text, nav background |
| `--copper` | `#C4622D` | Primary CTA, risk "high" state |
| `--slate-blue` | `#2E4057` | Section headers, card borders |
| `--sand` | `#F0E6D3` | Background wash, low-risk badge |
| `--sage` | `#4A7C59` | Success states, low-risk indicator |
| `--amber` | `#D4A017` | Warning / medium-risk state |
| `--mist` | `#F7F4F0` | Page background |
| `--rule` | `#D9D0C7` | Dividers, input borders |

**Rationale:** Copper + slate blue evokes trustworthy financial authority without the generic blue-fintech cliché. Sand and mist keep the feel warm and human — countering the cold anxiety that credit scoring can evoke.

### 2.2 Typography

| Role | Typeface | Weight | Size |
|------|----------|--------|------|
| Display | Playfair Display | 700 | 48–64px |
| Heading | DM Serif Display | 400 | 28–36px |
| UI Label | IBM Plex Mono | 500 | 12–14px |
| Body | Lora | 400 | 16px |
| Data / Number | IBM Plex Mono | 600 | 20–32px |

### 2.3 Component Library

- Inputs: rounded-sm (4px), `1px solid --rule`, focus ring in `--copper`
- Cards: `background: white`, `border: 1px solid --rule`, `box-shadow: 0 2px 12px rgba(13,27,42,0.06)`
- Buttons (Primary): `background: --copper`, white text, hover darkens 10%
- Buttons (Ghost): `border: 1.5px solid --copper`, `color: --copper`
- Risk Badges: pill shape, color-coded (sage/amber/copper)

---

## 3. Pages & Screens

### Page 1 — Landing / Hero (`/`)

**Purpose:** Establish credibility, communicate value proposition, drive users to the tool.

**Layout:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  LOGO  · Credit Risk Intelligence          [Docs]  [API]  [Login]   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────────────────┐    ┌─────────────────────────────┐  │
│   │                           │    │                             │  │
│   │  HERO TEXT (left 55%)     │    │  RELATABLE IMAGE            │  │
│   │                           │    │  (Financial professional    │  │
│   │  "Know Before You Lend."  │    │   reviewing analytics on    │  │
│   │                           │    │   a screen — warm office    │  │
│   │  Subheading: ML-powered   │    │   environment, diverse)     │  │
│   │  credit risk & limit      │    │                             │  │
│   │  scoring in seconds.      │    │                             │  │
│   │                           │    │                             │  │
│   │  [Try the Tool →]         │    │                             │  │
│   │  [View API Docs]          │    └─────────────────────────────┘  │
│   └───────────────────────────┘                                     │
│                                                                     │
│  ─────────────────── STATS STRIP ─────────────────────────────────  │
│   98.2% Accuracy   ·   <200ms Latency   ·   Open Source / MIT      │
├─────────────────────────────────────────────────────────────────────┤
│                   HOW IT WORKS (3 steps)                            │
│   [1. Submit Data]   →   [2. ML Scores Risk]   →   [3. Get Insight] │
├─────────────────────────────────────────────────────────────────────┤
│              FEATURE HIGHLIGHTS  (2-column card grid)               │
│  ┌──────────────────────┐   ┌──────────────────────────────────┐   │
│  │ Default Prediction   │   │ Credit Limit Recommendation      │   │
│  │ Binary classification│   │ Regression-based, adaptive       │   │
│  │ w/ probability score │   │ recommendation engine            │   │
│  └──────────────────────┘   └──────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                        FOOTER                                       │
│   GitHub · Email · MIT License                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Page 2 — Dashboard (`/dashboard`)

**Purpose:** Main tool interface — input customer data, view both predictions side by side.

**Layout:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  ← Back    CREDIT RISK DASHBOARD           [Health: ● Healthy]      │
├──────────────┬──────────────────────────────────────────────────────┤
│  SIDEBAR     │  MAIN CONTENT AREA                                   │
│  ─────────── │  ─────────────────────────────────────────────────── │
│  Navigation  │                                                       │
│  > Dashboard │  ┌──────────────────────────────────────────────┐   │
│    Default   │  │  INPUT FORM — Customer Financial Profile      │   │
│    Predict   │  │                                              │   │
│    Credit    │  │  Credit Limit: [___________]                  │   │
│    Limit     │  │  Age:          [___]                          │   │
│    History   │  │  Gender:       [ Male ▼ ]                     │   │
│    Settings  │  │  Education:    [ Graduate ▼ ]                 │   │
│  ─────────── │  │  Marital:      [ Married ▼ ]                  │   │
│              │  │  Pay Status:   [ Paid on time ▼ ]             │   │
│  API Status  │  │  Bill Amount:  [___________]                  │   │
│  ● Healthy   │  │  Payment Amt:  [___________]                  │   │
│              │  │                                               │   │
│              │  │  [Predict Default Risk]  [Predict Limit]      │   │
│              │  └──────────────────────────────────────────────┘   │
│              │                                                       │
│              │  ┌──────────────────┐  ┌──────────────────────────┐  │
│              │  │ DEFAULT RISK     │  │ CREDIT LIMIT             │  │
│              │  │ RESULT CARD      │  │ RESULT CARD              │  │
│              │  │                  │  │                          │  │
│              │  │  15%             │  │  ₦150,000                │  │
│              │  │  ● Low Risk      │  │  ↑ +50% from current     │  │
│              │  │                  │  │                          │  │
│              │  │  Risk Factors:   │  │  Recommendation Factors: │  │
│              │  │  · Low pay ratio │  │  · Good pay history      │  │
│              │  │                  │  │  · High pay/bill ratio   │  │
│              │  └──────────────────┘  └──────────────────────────┘  │
└──────────────┴──────────────────────────────────────────────────────┘
```

---

### Page 3 — Default Risk Detail (`/dashboard/default`)

**Purpose:** Dedicated view for default prediction with probability gauge.

**Layout:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  DEFAULT PREDICTION                                [Run New Analysis]│
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│     ┌───────────────────────────────────────────────────────┐      │
│     │                  RISK GAUGE                           │      │
│     │         ┌──────────────────────┐                      │      │
│     │         │    Semi-circle       │                      │      │
│     │         │    gauge: 0% – 100%  │                      │      │
│     │         │    Needle at 15%     │                      │      │
│     │         │    ● LOW RISK        │                      │      │
│     │         └──────────────────────┘                      │      │
│     │                  15.00%                               │      │
│     └───────────────────────────────────────────────────────┘      │
│                                                                     │
│     RISK FACTORS                    PARAMETERS USED                 │
│     ┌──────────────────────┐        ┌──────────────────────────┐    │
│     │ · Low pay/bill ratio │        │ Credit Limit: 100,000    │    │
│     │                      │        │ Age: 25                  │    │
│     │                      │        │ Payment Status: Paid     │    │
│     └──────────────────────┘        └──────────────────────────┘    │
│                                                                     │
│     RISK LEGEND                                                     │
│     [● 0–30% Low] [● 31–60% Medium] [● 61–100% High]               │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Page 4 — Credit Limit Detail (`/dashboard/credit-limit`)

**Purpose:** Dedicated view for credit limit recommendation.

**Layout:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  CREDIT LIMIT RECOMMENDATION                      [New Prediction]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RECOMMENDED LIMIT                                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │    Current: ████████████████░░░░░░░░░░░  100,000             │  │
│  │  Recommend: ████████████████████████████  150,000            │  │
│  │                          Adjustment: ×1.5                     │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  RECOMMENDATION FACTORS                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ✓ Good payment history          ✓ High payment to bill ratio│   │
│  │  ✓ Age factor positive           ✓ Education level positive  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ADJUSTMENT FACTOR BREAKDOWN                                        │
│  ┌────────────────────────────────────┐                             │
│  │  Bar chart: factors and weights   │                             │
│  └────────────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Page 5 — Prediction History (`/dashboard/history`)

**Purpose:** Log of past predictions for audit and trend analysis.

**Layout:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  PREDICTION HISTORY             [Filter ▼]  [Export CSV]            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Date       Type         Risk %   Limit        Status          │ │
│  │  ─────────────────────────────────────────────────────────     │ │
│  │  2026-05-03  Default      15%      —            ● Low Risk      │ │
│  │  2026-05-02  Credit Limit  —      150,000       ✓ Approved      │ │
│  │  2026-05-01  Default      72%      —            ● High Risk     │ │
│  │  2026-04-30  Default      38%      —            ● Medium Risk   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  [< Prev]  Page 1 of 4  [Next >]                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. User Flows

### Flow A — Predict Default Risk
```
Landing → [Try the Tool] → Dashboard → Fill Form → [Predict Default Risk]
→ Default Result Card (inline) → [View Full Detail] → Default Detail Page
```

### Flow B — Predict Credit Limit
```
Dashboard → Fill Form → [Predict Limit] → Credit Limit Card (inline)
→ [View Full Detail] → Credit Limit Detail Page
```

### Flow C — API Direct Access
```
Landing → [View API Docs] → External Docs / Swagger
```

---

## 5. API Integration Map

| UI Action | Endpoint | Method |
|-----------|----------|--------|
| Health status indicator | `/api/v1/health` | GET |
| Default prediction | `/api/v1/predict/default` | POST |
| Credit limit prediction | `/api/v1/predict/credit-limit` | POST |

**Error States:**
- API down → amber banner "Service temporarily unavailable"
- Validation error → inline field error messages
- Network timeout → toast notification with retry option

---

## 6. Responsive Breakpoints

| Breakpoint | Width | Notes |
|------------|-------|-------|
| Mobile | < 640px | Single column, stacked cards, hidden sidebar |
| Tablet | 640–1024px | Condensed sidebar, 1-col form |
| Desktop | > 1024px | Full layout as wireframed |

---

## 7. Accessibility Requirements

- WCAG 2.1 AA minimum
- All form inputs with visible labels (never placeholder-only)
- Risk colors always paired with text labels (not color-only)
- Keyboard navigable throughout
- Focus rings visible on all interactive elements

---

## 8. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Time to Interactive | < 2s on 4G |
| First Contentful Paint | < 1.2s |
| API response feedback | Loading state within 100ms |
| Bundle size | < 200KB gzipped |

---

*End of PRD*
