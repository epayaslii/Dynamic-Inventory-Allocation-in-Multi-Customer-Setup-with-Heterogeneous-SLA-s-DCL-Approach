"""
PDF report of the 2000-period simulation written by simulate_periods.py.
Needs matplotlib and reportlab (not part of the DynaPlex venv):
    python make_sim_report.py --prefix ../../docs/sim2000 --out ../../docs/simulation_2000_periods.pdf
"""
import argparse
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

C_NAMES = ["Customer 1", "Customer 2"]
COL = {"c1": "#1f5fa8", "c2": "#d1622b", "i1": "#1f5fa8", "i2": "#2a9d8f", "i3": "#c8892b", "grey": "#888888"}


def load(prefix, name):
    d = json.load(open(f"{prefix}_{name}.json"))
    return d["meta"], d["records"]


def fig_image(fig, width_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    w, h = PILImage.open(buf).size
    buf.seek(0)
    return Image(buf, width=width_mm * mm, height=width_mm * mm * h / w)


def arr(R, key):
    return np.array([r[key] for r in R])


def plots(meta, R, R2):
    n_c, n_i, T = meta["n_c"], meta["n_i"], meta["T"]
    t = arr(R, "t")
    out = []
    first = 3 * T  # zoom window: first three horizons

    # 1. Stock: OH' (before allocation) and OH_{t+1} (after), first three horizons.
    fig, axs = plt.subplots(n_i, 1, figsize=(9, 5.6), sharex=True)
    ohp, ohn, q = arr(R, "OH_prime"), arr(R, "OH_next"), arr(R, "order")
    for i in range(n_i):
        ax = axs[i]
        ax.step(t[:first], ohp[:first, i], where="mid", color=COL["grey"], lw=1, label="OH' (before allocation)")
        ax.step(t[:first], ohn[:first, i], where="mid", color=COL[f"i{i+1}"], lw=1.6, label="OH after allocation (holding base)")
        ax.set_ylabel(f"Item {i+1}\n(h={meta['h'][i]})")
        for k in range(1, 4):
            ax.axvline(k * T + 0.5, color="k", lw=0.5, ls=":")
    axs[0].legend(fontsize=7, ncol=2, loc="upper right")
    axs[-1].set_xlabel("period t   (dotted lines: horizon boundaries)")
    fig.suptitle("On-hand inventory, periods 1-60 (DCL)", fontsize=10)
    out.append(("On-hand inventory per item. The holding cost of a period is h_i times the black/colored line, i.e. the stock left after allocation.", fig_image(fig, 165)))

    # 2. Backorder per customer, end of period, and sawtooth B_ct vs allowance.
    e = arr(R, "e")
    B = arr(R, "B")
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    for c in range(n_c):
        axs[0].step(t[:first], e[:first, c], where="mid", color=COL[f"c{c+1}"], label=C_NAMES[c])
        axs[1].step(t[:first], B[:first, c], where="post", color=COL[f"c{c+1}"], label=f"{C_NAMES[c]}  B_ct")
        axs[1].axhline(meta["beta"][c], color=COL[f"c{c+1}"], ls="--", lw=0.9, label=f"beta = {meta['beta'][c]}")
    axs[0].set_ylabel("e_ct: units owed\nat end of period")
    axs[1].set_ylabel("B_ct: cumulative\nbackorder (unit-periods)")
    axs[1].set_xlabel("period t")
    for ax in axs:
        for k in range(1, 4):
            ax.axvline(k * T + 0.5, color="k", lw=0.5, ls=":")
    axs[0].legend(fontsize=7)
    axs[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Backorders and the SLA statistic, periods 1-60 (DCL)", fontsize=10)
    out.append(("End-of-period backorder e_ct (top) and its running sum B_ct within each horizon (bottom). B_ct resets to 0 at each dotted line; the dashed line is the allowance beta_c. The penalty is charged on whatever B_ct exceeds beta_c at the horizon's last period.", fig_image(fig, 165)))

    # 3. Cost per period, full 2000 periods: holding and penalty.
    hold = arr(R, "holding_cost")
    pen = arr(R, "penalty_cost")
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    axs[0].plot(t, hold, color=COL["i2"], lw=0.5)
    axs[0].set_ylabel("holding cost\nper period")
    ends = t[pen.sum(1) > 0]
    for c in range(n_c):
        axs[1].bar(t[R_end(R)], pen[R_end(R), c], width=T * 0.35, color=COL[f"c{c+1}"], label=C_NAMES[c], align="center")
    axs[1].set_ylabel("SLA penalty\n(horizon ends)")
    axs[1].set_xlabel("period t")
    axs[1].legend(fontsize=7)
    fig.suptitle("Cost components over all 2000 periods (DCL)", fontsize=10)
    out.append(("Holding cost is charged in every period (top); the SLA penalty only in the last period of each of the 100 horizons (bottom, one bar per customer per horizon).", fig_image(fig, 165)))

    # 4. B at horizon end per horizon vs beta.
    endsB = B[R_end(R)]
    endsB2 = arr(R2, "B")[R_end(R2)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.1), sharey=False)
    for c in range(n_c):
        h = np.arange(1, len(endsB) + 1)
        axs[c].plot(h, endsB[:, c], color=COL[f"c{c+1}"], lw=0.9, label="DCL")
        axs[c].plot(h, endsB2[:, c], color=COL["grey"], lw=0.7, alpha=0.8, label="Cost-greedy")
        axs[c].axhline(meta["beta"][c], color="k", ls="--", lw=0.9, label=f"beta = {meta['beta'][c]}")
        axs[c].set_title(C_NAMES[c], fontsize=9)
        axs[c].set_xlabel("horizon r")
        axs[c].set_ylabel("B_{c,rT}(T)")
        axs[c].legend(fontsize=7)
    out.append(("Cumulative backorder at the end of each of the 100 horizons against the allowance.", fig_image(fig, 165)))

    # 5. Cumulative cost.
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for Rx, lab, colr in ((R, "DCL", COL["c1"]), (R2, "Cost-greedy", COL["grey"])):
        ax.plot(arr(Rx, "t"), np.cumsum(arr(Rx, "cost")) / arr(Rx, "t"), color=colr, label=lab)
    ax.set_xlabel("period t")
    ax.set_ylabel("running average cost / period")
    ax.legend(fontsize=7)
    out.append(("Running average cost per period over the 2000 periods.", fig_image(fig, 165)))
    return out


def R_end(R):
    return np.array([r["R"] == 0 for r in R])


def summary_rows(meta, R):
    n_c = meta["n_c"]
    T = len(R)
    hold = sum(r["holding_cost"] for r in R)
    pen = [sum(r["penalty_cost"][c] for r in R) for c in range(n_c)]
    ends = [r for r in R if r["R"] == 0]
    over = [sum(1 for r in ends if r["B"][c] > meta["beta"][c]) for c in range(n_c)]
    meanB = [np.mean([r["B"][c] for r in ends]) for c in range(n_c)]
    excess = [np.mean([max(0, r["B"][c] - meta["beta"][c]) for r in ends]) for c in range(n_c)]
    return {
        "hold": hold / T, "pen": sum(pen) / T, "total": (hold + sum(pen)) / T,
        "pen_c": [p / T for p in pen], "over": over, "n_h": len(ends), "meanB": meanB, "excess": excess,
        "periods_hold_pos": sum(1 for r in R if r["holding_cost"] > 0),
        "periods_bo_pos": sum(1 for r in R if sum(r["e"]) > 0),
        "holds": sum(r["holds"] for r in R),
    }


def make_tables(R, meta):
    n_c, n_i = meta["n_c"], meta["n_i"]
    pairs = [f"{c+1}{i+1}" for c in range(n_c) for i in range(n_i)]
    # Table A: start-of-period state s_t.
    headA1 = ["", ""] + [""] * 1 + ["OH_it"] * n_i + ["q_{t-2}"] * n_i + ["q_{t-1}"] * n_i + ["b_cit (c,i)"] * 6 + ["d_cit (c,i)"] * 6 + ["B_{c,t-1}"] * 2
    headA2 = ["t", "r", "R_{t-1}"] + [f"i{i+1}" for i in range(n_i)] * 3 + pairs + pairs + ["c1", "c2"]
    rowsA = []
    for r in R:
        rowsA.append([r["t"], r["horizon"], r["R_prev"]] + r["OH"] + [r["pipeline"][i][0] for i in range(n_i)]
                     + [r["pipeline"][i][1] for i in range(n_i)] + r["b"] + r["d"] + r["B_prev"])
    # Table B: decision and outcome.
    headB1 = [""] + ["OH'_it"] * n_i + ["IP_it"] * n_i + ["q_it"] * n_i + ["a_cit (c,i)"] * 6 + ["OH_{i,t+1}"] * n_i + ["b_{ci,t+1} (c,i)"] * 6 + ["e_ct"] * 2 + ["B_ct"] * 2 + ["R_t", "hold", "penalty"]
    headB2 = ["t"] + [f"i{i+1}" for i in range(n_i)] * 3 + pairs + [f"i{i+1}" for i in range(n_i)] + pairs + ["c1", "c2", "c1", "c2", "", "cost", "cost"]
    rowsB = []
    for r in R:
        pen = sum(r["penalty_cost"])
        rowsB.append([r["t"]] + r["OH_prime"] + r["IP"] + r["order"] + r["a"] + r["OH_next"] + r["b_next"] + r["e"] + r["B"]
                     + [r["R"], f"{r['holding_cost']:.1f}", f"{pen:.0f}" if pen else "0"])
    return (headA1, headA2, rowsA), (headB1, headB2, rowsB)


def flow_table(head1, head2, rows, ends, font=5.6, col_w=None, repeat=2):
    data = [head1, head2] + [[str(x) for x in row] for row in rows]
    n = len(head1)
    t = Table(data, colWidths=col_w, repeatRows=repeat, rowHeights=None)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 0.8), ("RIGHTPADDING", (0, 0), (-1, -1), 0.8),
        ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#e6ebf2")),
        ("LINEBELOW", (0, 1), (-1, 1), 0.6, colors.black),
    ]
    # group separators in the header: span repeated labels
    start = 0
    for k in range(1, n + 1):
        if k == n or head1[k] != head1[start]:
            if head1[start] and k - start > 1:
                style.append(("SPAN", (start, 0), (k - 1, 0)))
            if k < n:
                style.append(("LINEBEFORE", (k, 0), (k, -1), 0.3, colors.HexColor("#999999")))
            start = k
    for idx, is_end in enumerate(ends):
        row = idx + 2
        if is_end:
            style.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#fde9c9")))
            style.append(("LINEBELOW", (0, row), (-1, row), 0.5, colors.HexColor("#c8892b")))
    t.setStyle(TableStyle(style))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="../../docs/sim2000")
    ap.add_argument("--out", default="../../docs/simulation_2000_periods.pdf")
    args = ap.parse_args()
    meta, R = load(args.prefix, "dcl")
    _, R2 = load(args.prefix, "costgreedy")
    n_c, n_i, T = meta["n_c"], meta["n_i"], meta["T"]
    S1, S2 = summary_rows(meta, R), summary_rows(meta, R2)

    ss = getSampleStyleSheet()
    body = ss["BodyText"]
    body.fontSize, body.leading = 9, 12
    h1, h2 = ss["Heading1"], ss["Heading2"]
    small = ss["BodyText"].clone("small", fontSize=8, leading=10, textColor=colors.HexColor("#444444"))

    port = Frame(18 * mm, 16 * mm, A4[0] - 36 * mm, A4[1] - 32 * mm, id="p")
    lw, lh = landscape(A4)
    land = Frame(8 * mm, 8 * mm, lw - 16 * mm, lh - 16 * mm, id="l", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = BaseDocTemplate(args.out, pagesize=A4, title="2000-period simulation of the multi-customer SLA model",
                          author="Eliz Payasli")
    doc.addPageTemplates([
        PageTemplate("port", frames=[port], pagesize=A4, onPage=lambda c, d: footer(c, d, A4)),
        PageTemplate("land", frames=[land], pagesize=landscape(A4), onPage=lambda c, d: footer(c, d, landscape(A4))),
    ])

    def footer(c, d, size):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.grey)
        c.drawRightString(size[0] - 10 * mm, 5 * mm, f"page {d.page}")
        c.restoreState()

    el = []
    el.append(Paragraph("2000-period simulation of the DCL allocation policy", h1))
    el.append(Paragraph("Multi-customer inventory allocation under heterogeneous SLAs: do holding cost and SLA penalty behave as the model says?", body))
    el.append(Spacer(1, 4 * mm))

    el.append(Paragraph("1. Set-up", h2))
    el.append(Paragraph(
        f"Instance of the paper's numerical section: |C| = {n_c} customers, |I| = {n_i} items, lead time tau = {meta['tau']}, review horizon T = {T}, "
        f"base stock S_i = {meta['S'][0]} for all items, h = {meta['h']}, beta = {meta['beta']}, p = {meta['p']}, Poisson demand with rates "
        "(customer 1: 1.0, 0.8, 0.5; customer 2: 0.7, 1.2, 0.6). The policy is the generation-3 DCL agent (trained in "
        "<font face='Courier'>dynaplex_runs/.../exp_a2b3c9be/agent_gen3</font>). The run starts from the paper's initial state "
        f"(OH_i1 = S_i, empty pipeline, no backorders, B = 0, R = T) with no warm-up, and covers {len(R)} periods = {len(R)//T} complete review horizons "
        f"(seed {meta['seed']}). The same demand stream is also run under cost-greedy, the best rule-based benchmark, for comparison.", body))

    el.append(Spacer(1, 2 * mm))
    rates = [1.0, 0.8, 0.5, 0.7, 1.2, 0.6]  # flattened c * |I| + i, as in simulate_periods.build_mdp
    prm = [["Size of the instance", ""],
           ["Customers |C|", n_c], ["Items |I|", n_i], ["Lead time tau (periods)", meta["tau"]],
           ["Review horizon T (periods)", T], ["Simulated periods", f"{len(R)} ({len(R)//T} horizons)"],
           ["", ""],
           ["Per item", "Item 1 / Item 2 / Item 3"],
           ["Holding cost h_i", " / ".join(str(x) for x in meta["h"])],
           ["Base-stock level S_i", " / ".join(str(x) for x in meta["S"])],
           ["", ""],
           ["Per customer", "Customer 1 / Customer 2"],
           ["Backorder allowance beta_c (unit-periods)", " / ".join(str(x) for x in meta["beta"])],
           ["Penalty cost p_c", " / ".join(str(x) for x in meta["p"])],
           ["", ""],
           ["Poisson demand rate per period", "Item 1 / Item 2 / Item 3"],
           ["  Customer 1", " / ".join(str(x) for x in rates[:n_i])],
           ["  Customer 2", " / ".join(str(x) for x in rates[n_i:])]]
    tp = Table([[str(x) for x in r] for r in prm], colWidths=[85 * mm, 70 * mm])
    bold = [(0, r) for r in (0, 7, 11, 15)]
    tp.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                            ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black)]
                           + [("FONTNAME", (0, r), (-1, r), "Helvetica-Bold") for _, r in bold]
                           + [("LINEBELOW", (0, r), (-1, r), 0.4, colors.black) for _, r in bold]))
    el.append(tp)

    el.append(Paragraph("2. Check against the model", h2))
    el.append(Paragraph(
        "In every one of the 2000 periods the simulation re-derives the following from the recorded values with the paper's formulas and asserts that they "
        "equal what the MDP code produced (script <font face='Courier'>simulate_periods.py</font>): "
        "OH'<sub>it</sub> = OH<sub>it</sub> + q<sub>i,t-tau</sub>; IP<sub>it</sub> and the order q<sub>it</sub> = (S<sub>i</sub> - IP<sub>it</sub>)<super>+</super>; "
        "allocation feasibility (a<sub>cit</sub> &le; b<sub>cit</sub> + d<sub>cit</sub>, &Sigma;<sub>c</sub> a<sub>cit</sub> &le; OH'<sub>it</sub>); "
        "OH<sub>i,t+1</sub> and b<sub>ci,t+1</sub>; the end-of-period backorder e<sub>ct</sub>; the horizon statistics B<sub>ct</sub> = B<sub>c,t-1</sub> + e<sub>ct</sub> and R<sub>t</sub>, "
        "including the reset to (0, ..., 0, T) after the last period of a horizon; and the period cost = holding cost on post-allocation stock + SLA penalty "
        "at horizon ends only. All assertions passed, for both policies.", body))

    el.append(Paragraph("3. Results", h2))
    rows = [["", "DCL (gen 3)", "Cost-greedy"],
            ["Average cost per period", f"{S1['total']:.2f}", f"{S2['total']:.2f}"],
            ["  holding cost per period", f"{S1['hold']:.2f}", f"{S2['hold']:.2f}"],
            ["  SLA penalty per period", f"{S1['pen']:.2f}", f"{S2['pen']:.2f}"],
            ["    of which customer 1", f"{S1['pen_c'][0]:.2f}", f"{S2['pen_c'][0]:.2f}"],
            ["    of which customer 2", f"{S1['pen_c'][1]:.2f}", f"{S2['pen_c'][1]:.2f}"],
            ["Horizons with penalty, customer 1 (of 100)", S1["over"][0], S2["over"][0]],
            ["Horizons with penalty, customer 2 (of 100)", S1["over"][1], S2["over"][1]],
            ["Mean B at horizon end, customer 1 (beta = 6)", f"{S1['meanB'][0]:.1f}", f"{S2['meanB'][0]:.1f}"],
            ["Mean B at horizon end, customer 2 (beta = 8)", f"{S1['meanB'][1]:.1f}", f"{S2['meanB'][1]:.1f}"],
            ["Periods with holding cost > 0", S1["periods_hold_pos"], S2["periods_hold_pos"]],
            ["Periods ending with some backorder", S1["periods_bo_pos"], S2["periods_bo_pos"]],
            ["HOLD actions taken", S1["holds"], S2["holds"]]]
    tb = Table([[str(x) for x in r] for r in rows], colWidths=[85 * mm, 35 * mm, 35 * mm])
    tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                            ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    el.append(tb)
    el.append(Spacer(1, 3 * mm))
    el.append(Paragraph(
        "<b>Both cost components are active, but at different times.</b> Holding cost is positive in "
        f"{S1['periods_hold_pos']} of 2000 periods (stock is left over after allocation), and it varies from period to period with the pipeline arrivals and demand. "
        f"The SLA penalty is zero in {len(R) - (len(R)//T)} of the 2000 periods by construction and is charged only in the last period of a horizon.", body))
    el.append(Paragraph(
        "<b>Observations.</b> (1) In this instance the penalty dominates: about "
        f"{100 * S1['pen'] / S1['total']:.0f}% of the cost is SLA penalty, and customer 1 exceeds its allowance in {S1['over'][0]} of 100 horizons under DCL "
        f"(customer 2 in {S1['over'][1]}). Backorders are therefore the rule, not the exception, with S<sub>i</sub> = 6; the allowances (6 and 8 unit-periods per 20 periods) "
        "are tight relative to the backorder this base stock produces. (2) The holding cost is identical under DCL and cost-greedy "
        f"({S1['hold']:.2f}). This follows from the model: neither policy ever holds stock back, so the on-hand inventory trajectory depends only on the demand, "
        "not on who receives the units. Policies differ only in how the same backorder is split between the customers. "
        f"(3) DCL's gain over cost-greedy on this single run is {S2['total'] - S1['total']:.2f} per period ({100 * (S2['total'] - S1['total']) / S2['total']:.1f}%): it lowers the penalty of customer 2 "
        "at the expense of customer 1. This is one seed and no confidence interval; it is a trace, not a benchmark. "
        "(4) The DCL agent never chose HOLD, consistent with the unit-by-unit action design.", body))

    for cap, img in plots(meta, R, R2):
        el.append(img)
        el.append(Paragraph(cap, small))
        el.append(Spacer(1, 3 * mm))

    el.append(PageBreak())
    el.append(Paragraph("4. How to read the period flow tables", h2))
    el.append(Paragraph(
        "Appendix A lists the state s<sub>t</sub> at the allocation decision of every period (columns in the paper's order: OH<sub>it</sub>, the two pipeline orders "
        "q<sub>i,t-2</sub> and q<sub>i,t-1</sub>, b<sub>cit</sub>, d<sub>cit</sub>, and the horizon statistics B<sub>c,t-1</sub>, R<sub>t-1</sub>, plus the horizon index r; "
        "the allowances beta = (6, 8) are constant and are not repeated). Pairs 'ci' are customer c, item i. "
        "Appendix B lists what happens next: OH' and IP, the order q<sub>it</sub> placed this period, the allocation a<sub>cit</sub>, the resulting "
        "OH<sub>i,t+1</sub> and b<sub>ci,t+1</sub>, the end-of-period backorder e<sub>ct</sub>, the updated B<sub>ct</sub> and R<sub>t</sub>, and the two cost components of the period. "
        "Orange rows are the last period of a horizon: B<sub>ct</sub> is compared with beta<sub>c</sub> there, the penalty is charged, and the next row starts with B = 0, R = 20. "
        "Row t of Appendix A and row t of Appendix B describe the same period; OH<sub>i,t+1</sub> and b<sub>ci,t+1</sub> in B reappear as OH and b in row t+1 of A. "
        "The files <font face='Courier'>docs/sim2000_dcl.json</font> and <font face='Courier'>sim2000_costgreedy.json</font> contain the same records in machine-readable form.", body))

    (hA1, hA2, rowsA), (hB1, hB2, rowsB) = make_tables(R, meta)
    ends = [r["R"] == 0 for r in R]
    # Worked excerpt: the first horizon, with a few lines of commentary.
    el.append(NextPageTemplate("land"))
    el.append(PageBreak())
    wA = [7 * mm, 6 * mm, 9 * mm] + [8.5 * mm] * 3 * n_i + [9 * mm] * 12 + [10 * mm] * 2
    el.append(Paragraph("Appendix A. State s<sub>t</sub> at the allocation decision, periods 1-2000 (DCL)", h2))
    chunk = 37
    for k in range(0, len(R), chunk):
        el.append(flow_table(hA1, hA2, rowsA[k:k + chunk], ends[k:k + chunk], font=6, col_w=wA))
        el.append(PageBreak())
    el.append(Paragraph("Appendix B. Decisions and outcome of each period, periods 1-2000 (DCL)", h2))
    wB = [7 * mm] + [7.5 * mm] * 9 + [8 * mm] * 6 + [8.5 * mm] * 3 + [8 * mm] * 6 + [8 * mm] * 4 + [7 * mm, 9 * mm, 10 * mm]
    for k in range(0, len(R), chunk):
        el.append(flow_table(hB1, hB2, rowsB[k:k + chunk], ends[k:k + chunk], font=6, col_w=wB))
        if k + chunk < len(R):
            el.append(PageBreak())
    doc.build(el)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
