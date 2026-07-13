const $ = (selector) => document.querySelector(selector)
const status = $("#status")
let current = null
let queue = []

async function api(url, options) {
  const response = await fetch(url, options)
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`)
  return response.json()
}

function show(tab) {
  for (const section of document.querySelectorAll("main > section")) section.hidden = section.id !== tab
}

document.querySelectorAll("[data-tab]").forEach((button) => button.onclick = () => show(button.dataset.tab))
document.querySelectorAll("[data-pipeline]").forEach((button) => button.onclick = async () => {
  try {
    const result = await api("/api/pipeline-run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: button.dataset.pipeline, limit: Number($("#pipeline-limit").value), workers: Number($("#pipeline-workers").value), model: $("#pipeline-model").value }) })
    $("#pipeline-output").textContent = (result.job.output || []).join("\n")
  } catch (error) { status.textContent = error.message }
})

$("#load-review").onclick = async () => {
  const mode = $("#candidate-mode").value
  const endpoint = mode === "links" ? "/api/link-candidates?limit=1" : mode === "teacher" ? null : "/api/crawler-candidates?limit=1"
  if (!endpoint) { status.textContent = "Load a teacher batch from Data files first."; return }
  try {
    const [item] = await api(endpoint)
    current = item || null
    $("#item").innerHTML = !item ? "No items." : `<a href="${item.target_url || item.url}" target="_blank" rel="noreferrer">${item.target_url || item.url}</a><p>${item.anchor || item.title || ""}</p><p>${item.snippet || ""}</p><p>${[1,2,3,4,5].map((rating) => `<button data-rating="${rating}">${rating}</button>`).join("")}</p>`
  } catch (error) { status.textContent = error.message }
}

$("#item").onclick = async (event) => {
  const button = event.target.closest("[data-rating]")
  if (!button || !current) return
  const link = $("#candidate-mode").value === "links"
  try {
    await api(link ? "/api/link-rating" : "/api/rating", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(link ? { link_id: current.id, rating: Number(button.dataset.rating) } : { result_id: current.id, rating: Number(button.dataset.rating) }) })
    $("#load-review").click()
  } catch (error) { status.textContent = error.message }
}

$("#evaluate-run").onclick = async () => {
  const snapshot = $("#snapshot").value
  try {
    const result = await api("/api/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ snapshot, baseline_page: $("#baseline-page").value, candidate_page: $("#candidate-page").value, baseline_link: $("#baseline-link").value, candidate_link: $("#candidate-link").value }) })
    $("#evaluation").textContent = JSON.stringify(result.report, null, 2)
    const report = document.createElement("a")
    report.href = `/api/evaluate/report?snapshot=${encodeURIComponent(snapshot)}`
    report.textContent = "Open report"
    $("#evaluation").append(" ", report)
    queue = result.queue || []
    renderQueue()
    $("#decision").hidden = false
  } catch (error) { status.textContent = error.message }
}

function renderQueue() {
  const item = queue.find((entry) => !entry.rated)
  $("#queue").textContent = item ? `Review queue: ${queue.filter((entry) => entry.rated).length}/${queue.length}. ${item.kind}: ${item.item.target_url || item.item.url || ""}` : `Review queue: ${queue.length}/${queue.length}.`
  if (!item) return
  for (const rating of [1, 2, 3, 4, 5]) {
    const button = document.createElement("button")
    button.textContent = rating
    button.onclick = async () => {
      try {
        await api("/api/evaluate/rating", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ snapshot: $("#snapshot").value, item_id: item.id, rating }) })
        item.rated = true; renderQueue()
      } catch (error) { status.textContent = error.message }
    }
    $("#queue").append(" ", button)
  }
}

$("#live-smoke").onclick = async () => {
  try {
    const result = await api("/api/evaluate/live-smoke", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ baseline_page: $("#baseline-page").value, candidate_page: $("#candidate-page").value, baseline_link: $("#baseline-link").value, candidate_link: $("#candidate-link").value }) })
    status.textContent = `Live smoke: ${result.output}`
  } catch (error) { status.textContent = error.message }
}

$("#decision").onclick = async (event) => {
  const button = event.target.closest("[data-decision]")
  if (!button) return
  try { await api("/api/evaluate/decision", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ snapshot: $("#snapshot").value, decision: button.dataset.decision }) }); status.textContent = "Decision recorded. Artifacts were not changed." } catch (error) { status.textContent = error.message }
}

setInterval(async () => { try { const result = await api("/api/pipeline-status"); $("#pipeline-output").textContent = (result.job.output || ["No job started."]).join("\n") } catch {} }, 1200)
