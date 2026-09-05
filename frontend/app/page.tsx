
"use client";

import { useEffect, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type RiskDecision = {
  transaction_id: string;
  risk_score: number;
  risk_level: string;
  action: string;
  risk_signals: string[];
  decision_reason: string;
  model_version: string;
  policy_version: string;
  decision_timestamp?: string;
};

type DashboardData = {
  total_transactions: number;
  high_risk: number;
  medium_risk: number;
  low_risk: number;
  recent_decisions: RiskDecision[];
};

type PendingReview = {
  review_id: number;
  transaction_id: string;
  risk_decision_id: number;
  status: string;
  risk_score: number;
  risk_level: string;
  action: string;
  risk_signals: string[];
  decision_reason: string;
  reviewer?: string | null;
  review_reason?: string | null;
  created_at: string;
};

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [reviews, setReviews] = useState<PendingReview[]>([]);

  const [pageError, setPageError] = useState("");
  const [message, setMessage] = useState("");

  const [transactionId, setTransactionId] = useState("");
  const [customerId, setCustomerId] = useState("");
  const [paymentMethod, setPaymentMethod] = useState("");
  const [amount, setAmount] = useState("");

  const [refundRequested, setRefundRequested] = useState(false);
  const [refundAmount, setRefundAmount] = useState("");
  const [refundDelayHours, setRefundDelayHours] = useState("");
  const [previousOrders, setPreviousOrders] = useState("");
  const [previousRefunds, setPreviousRefunds] = useState("");
  const [historicalRefundRate, setHistoricalRefundRate] = useState("");
  const [highValueTransaction, setHighValueTransaction] = useState(false);
  const [rapidRefund, setRapidRefund] = useState(false);
  const [refundAmountRatio, setRefundAmountRatio] = useState("");
  const [refundFrequencySignal, setRefundFrequencySignal] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [reviewingId, setReviewingId] = useState<number | null>(null);

  const [reviewer, setReviewer] = useState("");
  const [reviewReason, setReviewReason] = useState("");

  async function getApiError(response: Response, fallback: string) {
    try {
      const body = await response.json();

      if (typeof body?.detail === "string") {
        return body.detail;
      }

      if (Array.isArray(body?.detail)) {
        return body.detail
          .map((item: any) => item?.msg || JSON.stringify(item))
          .join("; ");
      }

      if (body?.message) {
        return body.message;
      }
    } catch {
      // Ignore JSON parsing errors.
    }

    return `${fallback} (${response.status})`;
  }

  async function loadDashboard() {
    const response = await fetch(`${API_BASE}/dashboard/summary`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(
        await getApiError(response, "Failed to fetch dashboard data.")
      );
    }

    const dashboardData = await response.json();
    setData(dashboardData);
  }

  async function loadPendingReviews() {
    const response = await fetch(`${API_BASE}/reviews/pending`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(
        await getApiError(response, "Failed to fetch pending reviews.")
      );
    }

    const reviewData = await response.json();
    setReviews(reviewData);
  }

  async function loadDashboardData() {
    try {
      setPageError("");

      await Promise.all([
        loadDashboard(),
        loadPendingReviews(),
      ]);
    } catch (error) {
      setPageError(
        error instanceof Error
          ? error.message
          : "Could not connect to RazorGuard API."
      );
    }
  }

  useEffect(() => {
    loadDashboardData();

    const interval = window.setInterval(() => {
      loadDashboardData();
    }, 10000);

    return () => window.clearInterval(interval);
  }, []);

  function resetForm() {
    setTransactionId("");
    setCustomerId("");
    setPaymentMethod("");
    setAmount("");
    setRefundRequested(false);
    setRefundAmount("");
    setRefundDelayHours("");
    setPreviousOrders("");
    setPreviousRefunds("");
    setHistoricalRefundRate("");
    setHighValueTransaction(false);
    setRapidRefund(false);
    setRefundAmountRatio("");
    setRefundFrequencySignal("");
  }

  async function submitTransaction(
    e: React.FormEvent<HTMLFormElement>
  ) {
    e.preventDefault();

    if (submitting) {
      return;
    }

    setMessage("");
    setPageError("");

    const cleanTransactionId = transactionId.trim();
    const cleanCustomerId = customerId.trim();

    if (!cleanTransactionId) {
      setMessage("Transaction ID is required.");
      return;
    }

    if (!cleanCustomerId) {
      setMessage("Customer ID is required.");
      return;
    }

    const numericAmount = Number(amount);
    const numericRefundAmount = Number(refundAmount || 0);
    const numericRefundDelayHours = Number(refundDelayHours || 0);
    const numericPreviousOrders = Number(previousOrders || 0);
    const numericPreviousRefunds = Number(previousRefunds || 0);
    const numericHistoricalRefundRate = Number(
      historicalRefundRate || 0
    );
    const numericRefundAmountRatio = Number(
      refundAmountRatio || 0
    );
    const numericRefundFrequencySignal = Number(
      refundFrequencySignal || 0
    );

    if (!Number.isFinite(numericAmount) || numericAmount < 0) {
      setMessage("Amount must be a valid non-negative number.");
      return;
    }

    if (
      !Number.isFinite(numericRefundAmount) ||
      numericRefundAmount < 0
    ) {
      setMessage("Refund amount must be a valid non-negative number.");
      return;
    }

    if (numericRefundAmount > numericAmount) {
      setMessage("Refund amount cannot exceed transaction amount.");
      return;
    }

    if (
      !refundRequested &&
      numericRefundAmount !== 0
    ) {
      setMessage(
        "Refund amount must be 0 when Refund Requested is not selected."
      );
      return;
    }

    if (
      numericHistoricalRefundRate < 0 ||
      numericHistoricalRefundRate > 1
    ) {
      setMessage("Historical refund rate must be between 0 and 1.");
      return;
    }

    if (
      numericRefundAmountRatio < 0 ||
      numericRefundAmountRatio > 1
    ) {
      setMessage("Refund amount ratio must be between 0 and 1.");
      return;
    }

    setSubmitting(true);

    const transaction = {
      transaction_id: cleanTransactionId,
      customer_id: cleanCustomerId,
      timestamp: new Date().toISOString(),
      amount: numericAmount,
      payment_method: paymentMethod,

      refund_requested: refundRequested,
      refund_amount: numericRefundAmount,
      refund_delay_hours: numericRefundDelayHours,

      previous_orders: numericPreviousOrders,
      previous_refunds: numericPreviousRefunds,
      historical_refund_rate: numericHistoricalRefundRate,

      high_value_transaction: highValueTransaction,
      rapid_refund: rapidRefund,
      refund_amount_ratio: numericRefundAmountRatio,
      refund_frequency_signal: numericRefundFrequencySignal,
    };

    try {
      /*
       * STEP 1
       * Persist the transaction.
       */
      const transactionResponse = await fetch(
        `${API_BASE}/transactions`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(transaction),
        }
      );

      if (!transactionResponse.ok) {
        throw new Error(
          await getApiError(
            transactionResponse,
            "Transaction ingestion failed."
          )
        );
      }

      /*
       * STEP 2
       * Run the ML risk engine.
       */
      const riskResponse = await fetch(
        `${API_BASE}/transactions/${encodeURIComponent(
          cleanTransactionId
        )}/risk`,
        {
          method: "POST",
          headers: {
            Accept: "application/json",
          },
        }
      );

      if (!riskResponse.ok) {
        throw new Error(
          await getApiError(
            riskResponse,
            "Risk evaluation failed."
          )
        );
      }

      const riskDecision: RiskDecision =
        await riskResponse.json();

      /*
       * STEP 3
       * Show the actual decision immediately.
       */
      setMessage(
        `Transaction evaluated: ${riskDecision.risk_level} risk ` +
          `(${riskDecision.risk_score.toFixed(4)}) — ` +
          `${riskDecision.action}.`
      );

      /*
       * STEP 4
       * Refresh dashboard and review queue.
       */
      await loadDashboardData();

      /*
       * STEP 5
       * Clear the form only after successful processing.
       */
      resetForm();
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Failed to process transaction."
      );
    } finally {
      setSubmitting(false);
    }
  }

  async function reviewCase(
    reviewId: number,
    status: "CONFIRMED_ABUSE" | "FALSE_POSITIVE"
  ) {
    if (!reviewer.trim()) {
      setMessage("Please enter a reviewer name.");
      return;
    }

    if (!reviewReason.trim()) {
      setMessage("Please enter a review reason.");
      return;
    }

    setReviewingId(reviewId);
    setMessage("");
    setPageError("");

    try {
      const response = await fetch(
        `${API_BASE}/reviews/${reviewId}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({
            status,
            reviewer: reviewer.trim(),
            review_reason: reviewReason.trim(),
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          await getApiError(
            response,
            "Failed to update review."
          )
        );
      }

      setMessage(
        status === "CONFIRMED_ABUSE"
          ? "Review marked as confirmed abuse."
          : "Review marked as false positive."
      );

      setReviewer("");
      setReviewReason("");

      await loadDashboardData();
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Failed to update review."
      );
    } finally {
      setReviewingId(null);
    }
  }

  if (!data) {
    return (
      <main className="min-h-screen bg-[#070a12] text-white flex items-center justify-center px-6">
        <div className="text-center">
          <div className="mx-auto mb-5 h-10 w-10 animate-spin rounded-full border-2 border-white/15 border-t-indigo-400" />
          <p className="text-sm text-white/50">Connecting to RazorGuard...</p>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[#05070d] text-white">
      <div className="relative mx-auto max-w-[1550px] px-5 py-6 sm:px-8 lg:px-10">
        <div className="pointer-events-none fixed inset-0 -z-0 overflow-hidden">
          <div className="absolute left-[8%] top-[-180px] h-[520px] w-[520px] rounded-full bg-indigo-600/15 blur-[140px]" />
          <div className="absolute right-[-120px] top-[18%] h-[430px] w-[430px] rounded-full bg-cyan-500/10 blur-[130px]" />
          <div className="absolute bottom-[-180px] left-[35%] h-[480px] w-[480px] rounded-full bg-violet-600/10 blur-[150px]" />
        </div>
        <header className="flex flex-col gap-5 border-b border-white/10 pb-7 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-white to-indigo-200 text-lg font-black text-[#070a12] shadow-[0_0_35px_rgba(129,140,248,0.25)]">
              R
              <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full bg-emerald-400 ring-4 ring-[#05070d]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">RazorGuard AI</h1>
              <p className="text-xs text-white/40">Merchant Risk Intelligence</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-xs font-semibold text-emerald-300 shadow-[0_0_25px_rgba(52,211,153,0.08)]">
              <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.9)]" />
              Risk Engine Operational
            </div>
            <button onClick={loadDashboardData} className="rounded-xl border border-white/10 bg-white/5 px-5 py-2.5 text-sm font-medium hover:bg-white/10">
              Refresh
            </button>
          </div>
        </header>

        {pageError && (
          <div className="mt-5 rounded-2xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-300">{pageError}</div>
        )}
        {message && (
          <div className="mt-5 rounded-2xl border border-indigo-400/20 bg-indigo-400/10 p-4 text-sm text-indigo-200">{message}</div>
        )}

        <section className="relative py-12 lg:py-16">
          <div className="absolute right-0 top-8 hidden w-[330px] rounded-3xl border border-white/10 bg-white/[0.035] p-5 backdrop-blur-xl lg:block">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/30">Live Protection</span>
              <span className="text-[10px] font-mono text-emerald-300">ACTIVE</span>
            </div>
            <div className="mt-5 flex items-center gap-4">
              <div className="relative h-16 w-16 rounded-full border border-indigo-400/30 bg-indigo-400/10 p-1">
                <div className="flex h-full w-full items-center justify-center rounded-full border border-white/10 text-sm font-bold">AI</div>
              </div>
              <div>
                <p className="text-sm font-semibold">Behavioral Risk Engine</p>
                <p className="mt-1 text-xs text-white/35">Continuous transaction analysis</p>
              </div>
            </div>
          </div>
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.22em] text-indigo-400">Risk Command Center</p>
          <h2 className="max-w-3xl text-4xl font-black tracking-[-0.04em] sm:text-6xl">
            Detect risky behavior before it becomes <span className="bg-gradient-to-r from-indigo-300 via-cyan-300 to-indigo-400 bg-clip-text text-transparent">merchant loss.</span>
          </h2>
          <p className="mt-4 max-w-2xl text-base leading-7 text-white/45">
            AI-powered refund risk detection with explainable decisions, controlled actions, human review, and auditability.
          </p>
        </section>

        <section className="relative grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Total Transactions" value={data.total_transactions} subtitle="Processed by RazorGuard" />
          <MetricCard label="High Risk" value={data.high_risk} subtitle="Requires human attention" emphasis="danger" />
          <MetricCard label="Medium Risk" value={data.medium_risk} subtitle="Monitor behavior" emphasis="warning" />
          <MetricCard label="Low Risk" value={data.low_risk} subtitle="No immediate concern" emphasis="safe" />
        </section>

        <section className="mt-8 grid grid-cols-1 gap-6 xl:grid-cols-[1.65fr_0.85fr]">
          <div className="rounded-3xl border border-white/10 bg-white/[0.045] p-5 shadow-[0_25px_90px_rgba(0,0,0,0.22)] backdrop-blur-xl sm:p-6">
            <div className="flex flex-col gap-4 border-b border-white/10 pb-5 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="text-xl font-bold">Review Queue</h3>
                  <span className="rounded-full border border-red-400/20 bg-red-400/10 px-3 py-1 text-xs font-semibold text-red-300">{reviews.length} pending</span>
                </div>
                <p className="mt-1 text-sm text-white/40">High-risk transactions requiring human judgment.</p>
              </div>
              <span className="text-xs text-white/25">Human-in-the-loop</span>
            </div>

            <div className="mt-5 space-y-4">
              {reviews.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-white/10 p-10 text-center text-sm text-white/35">No pending reviews.</div>
              ) : reviews.map((review) => (
                <div key={review.review_id} className="group rounded-2xl border border-white/10 bg-black/25 p-5 shadow-inner shadow-white/[0.02] transition hover:border-indigo-400/25 hover:bg-white/[0.035]">
                  <div className="flex flex-col gap-6 xl:flex-row xl:justify-between">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="font-mono text-sm font-semibold">{review.transaction_id}</span>
                        <RiskBadge level={review.risk_level} />
                        <span className="rounded-full bg-white/5 px-2.5 py-1 text-[11px] text-white/40">{review.action}</span>
                      </div>

                      <div className="mt-5 flex items-end gap-4">
                        <div className="shrink-0">
                          <p className="text-[10px] font-semibold uppercase tracking-widest text-white/30">Risk Score</p>
                          <p className="mt-1 text-5xl font-black tracking-[-0.04em]">{(review.risk_score * 100).toFixed(1)}<span className="text-lg text-white/30">%</span></p>
                        </div>
                        <div className="mb-2 h-2 flex-1 overflow-hidden rounded-full bg-white/10">
                          <div className="h-full rounded-full bg-gradient-to-r from-red-500 via-orange-400 to-yellow-300 shadow-[0_0_16px_rgba(248,113,113,0.45)]" style={{ width: `${Math.min(review.risk_score * 100, 100)}%` }} />
                        </div>
                      </div>

                      <div className="mt-5 rounded-xl border border-white/5 bg-white/[0.025] p-4">
                        <p className="text-[10px] font-semibold uppercase tracking-widest text-white/30">Why RazorGuard flagged it</p>
                        <p className="mt-2 text-sm leading-6 text-white/75">{review.decision_reason}</p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {review.risk_signals?.map((signal) => (
                            <span key={signal} className="rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs text-white/50">{signal}</span>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="w-full xl:w-[280px] xl:pt-1">
                      <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/30">Reviewer Decision</p>
                      <div className="space-y-2">
                        <input className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-white/25 outline-none focus:border-indigo-400/50" placeholder="Reviewer name" value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
                        <input className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-white/25 outline-none focus:border-indigo-400/50" placeholder="Review reason" value={reviewReason} onChange={(e) => setReviewReason(e.target.value)} />
                        <div className="grid grid-cols-2 gap-2 pt-1">
                          <button type="button" disabled={reviewingId === review.review_id} onClick={() => reviewCase(review.review_id, "CONFIRMED_ABUSE")} className="rounded-xl bg-white px-3 py-3 text-xs font-bold text-black hover:bg-red-100 disabled:opacity-40">
                            {reviewingId === review.review_id ? "Saving..." : "Confirm Abuse"}
                          </button>
                          <button type="button" disabled={reviewingId === review.review_id} onClick={() => reviewCase(review.review_id, "FALSE_POSITIVE")} className="rounded-xl border border-white/10 bg-white/5 px-3 py-3 text-xs font-semibold hover:bg-white/10 disabled:opacity-40">
                            False Positive
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="relative overflow-hidden rounded-3xl border border-indigo-400/20 bg-gradient-to-b from-indigo-500/[0.14] via-white/[0.04] to-white/[0.02] p-6 shadow-[0_25px_90px_rgba(79,70,229,0.10)] backdrop-blur-xl">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-400">Intelligence Layer</p>
            <h3 className="mt-2 text-2xl font-bold">How RazorGuard decides</h3>
            <p className="mt-2 text-sm leading-6 text-white/40">Risk is estimated by ML, controlled by deterministic policy, then routed to a human when scrutiny is required.</p>

            <div className="mt-7 space-y-3">
              <PipelineStep number="01" title="Behavioral Features" text="Refund history, frequency, amount ratio and transaction context." />
              <PipelineStep number="02" title="ML Risk Model" text="Logistic regression produces a continuous risk score." />
              <PipelineStep number="03" title="Policy Engine" text="Risk score becomes LOW, MEDIUM or HIGH." />
              <PipelineStep number="04" title="Human Review" text="High-risk cases are routed to merchant reviewers." />
            </div>

            <div className="mt-7 rounded-2xl border border-white/10 bg-black/20 p-4">
              <div className="flex justify-between text-xs"><span className="text-white/40">Model</span><span className="font-mono text-white/70">logistic_v1</span></div>
              <div className="mt-3 flex justify-between text-xs"><span className="text-white/40">Policy</span><span className="font-mono text-white/70">policy_v1</span></div>
            </div>
          </div>
        </section>

        <section className="mt-8 rounded-3xl border border-white/10 bg-white/[0.035] p-6 md:p-8">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-400">Live Evaluation</p>
          <h3 className="mt-2 text-2xl font-bold">Evaluate a Transaction</h3>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-white/40">Send transaction behavior through the same risk pipeline used by RazorGuard.</p>

          <form onSubmit={submitTransaction} className="mt-7 grid grid-cols-1 gap-4 md:grid-cols-2">
            <Input placeholder="Transaction ID" value={transactionId} onChange={(e) => setTransactionId(e.target.value)} required />
            <Input placeholder="Customer ID" value={customerId} onChange={(e) => setCustomerId(e.target.value)} required />
            <select className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none focus:border-indigo-400/50" value={paymentMethod} onChange={(e) => setPaymentMethod(e.target.value)} required>
              <option value="" className="bg-[#070a12]">Payment Method</option>
              <option value="card" className="bg-[#070a12]">Card</option>
              <option value="upi" className="bg-[#070a12]">UPI</option>
              <option value="netbanking" className="bg-[#070a12]">Net Banking</option>
              <option value="wallet" className="bg-[#070a12]">Wallet</option>
            </select>
            <Input type="number" min="0" step="0.01" placeholder="Transaction Amount" value={amount} onChange={(e) => setAmount(e.target.value)} required />
            <Input type="number" min="0" step="0.01" placeholder="Refund Amount" value={refundAmount} onChange={(e) => setRefundAmount(e.target.value)} />
            <Input type="number" min="0" step="0.01" placeholder="Refund Delay (hours)" value={refundDelayHours} onChange={(e) => setRefundDelayHours(e.target.value)} />
            <Input type="number" min="0" placeholder="Previous Orders" value={previousOrders} onChange={(e) => setPreviousOrders(e.target.value)} />
            <Input type="number" min="0" placeholder="Previous Refunds" value={previousRefunds} onChange={(e) => setPreviousRefunds(e.target.value)} />
            <Input type="number" min="0" max="1" step="0.01" placeholder="Historical Refund Rate (0–1)" value={historicalRefundRate} onChange={(e) => setHistoricalRefundRate(e.target.value)} />
            <Input type="number" min="0" max="1" step="0.01" placeholder="Refund Amount Ratio (0–1)" value={refundAmountRatio} onChange={(e) => setRefundAmountRatio(e.target.value)} />
            <Input type="number" min="0" step="1" placeholder="Refund Frequency Signal" value={refundFrequencySignal} onChange={(e) => setRefundFrequencySignal(e.target.value)} />

            <div className="flex flex-wrap items-center gap-5 rounded-xl border border-white/10 bg-white/5 px-4 py-3">
              <Toggle label="Refund Requested" checked={refundRequested} onChange={setRefundRequested} />
              <Toggle label="High Value" checked={highValueTransaction} onChange={setHighValueTransaction} />
              <Toggle label="Rapid Refund" checked={rapidRefund} onChange={setRapidRefund} />
            </div>

            <button type="submit" disabled={submitting} className="rounded-xl bg-white px-6 py-3.5 text-sm font-bold text-black hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-40 md:col-span-2">
              {submitting ? "Running Risk Engine..." : "Evaluate Transaction"}
            </button>
          </form>
        </section>

        <section className="mt-8 pb-12">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/30">Decision Log</p>
          <h3 className="mt-1 text-2xl font-bold">Recent Risk Decisions</h3>

          <div className="mt-5 overflow-hidden rounded-3xl border border-white/10 bg-white/[0.035]">
            {data.recent_decisions.length === 0 ? (
              <div className="p-8 text-center text-sm text-white/30">No risk decisions yet.</div>
            ) : (
              <div className="divide-y divide-white/10">
                {data.recent_decisions.map((decision) => (
                  <div key={`${decision.transaction_id}-${decision.decision_timestamp ?? decision.risk_score}`} className="grid gap-5 p-5 md:grid-cols-[1.3fr_0.6fr_0.8fr_1.5fr] md:items-center">
                    <div>
                      <p className="font-mono text-sm font-semibold">{decision.transaction_id}</p>
                      <p className="mt-1 text-xs text-white/30">{decision.model_version}</p>
                    </div>
                    <div>
                      <p className="text-[10px] uppercase tracking-wider text-white/30">Risk</p>
                      <p className="mt-1 text-xl font-bold">{(decision.risk_score * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <RiskBadge level={decision.risk_level} />
                      <p className="mt-2 text-xs text-white/40">{decision.action}</p>
                    </div>
                    <div>
                      <p className="text-sm text-white/65">{decision.decision_reason}</p>
                      <p className="mt-2 text-xs leading-5 text-white/30">{decision.risk_signals?.join(" • ") || "No specific signals"}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function MetricCard({
  label,
  value,
  subtitle,
  emphasis = "default",
}: {
  label: string;
  value: number;
  subtitle: string;
  emphasis?: "default" | "danger" | "warning" | "safe";
}) {
  const valueClass = {
    default: "text-white",
    danger: "text-red-300",
    warning: "text-amber-300",
    safe: "text-emerald-300",
  }[emphasis];

  const dotClass = {
    default: "bg-indigo-400",
    danger: "bg-red-400",
    warning: "bg-amber-400",
    safe: "bg-emerald-400",
  }[emphasis];

  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.035] p-6 hover:-translate-y-0.5 hover:border-white/20">
      <div className="flex items-start justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-white/35">{label}</p>
        <span className={`h-2 w-2 rounded-full ${dotClass}`} />
      </div>
      <p className={`mt-6 text-4xl font-bold tracking-tight ${valueClass}`}>{value}</p>
      <p className="mt-2 text-xs text-white/30">{subtitle}</p>
    </div>
  );
}

function PipelineStep({ number, title, text }: { number: string; title: string; text: string }) {
  return (
    <div className="flex gap-4 rounded-2xl border border-white/5 bg-black/20 p-4">
      <div className="font-mono text-xs text-indigo-400">{number}</div>
      <div>
        <p className="text-sm font-semibold">{title}</p>
        <p className="mt-1 text-xs leading-5 text-white/35">{text}</p>
      </div>
    </div>
  );
}

function Input({ type = "text", ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input type={type} className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-white/25 outline-none focus:border-indigo-400/50" {...props} />;
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-white/55">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="h-4 w-4 accent-indigo-500" />
      {label}
    </label>
  );
}

function RiskBadge({ level }: { level: string }) {
  if (level === "HIGH") {
    return <span className="rounded-full border border-red-400/20 bg-red-400/10 px-3 py-1 text-xs font-bold text-red-300">HIGH</span>;
  }
  if (level === "MEDIUM") {
    return <span className="rounded-full border border-amber-400/20 bg-amber-400/10 px-3 py-1 text-xs font-bold text-amber-300">MEDIUM</span>;
  }
  return <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-bold text-emerald-300">LOW</span>;
}
