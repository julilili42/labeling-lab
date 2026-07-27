const $ = (selector) => document.querySelector(selector)
const status = $("#status")
let current = null
let reviewBatch = []
let reviewPath = ""

function updateReviewCounter() {
  const teacher = $("#candidate-mode").value === "teacher"
  $("#review-counter").textContent = teacher && reviewPath ? `${reviewBatch.length + Number(Boolean(current))} remaining` : ""
}

async function api(url, options) {
  const response = await fetch(url, options)
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`)
  return response.json()
}

function show(tab) {
  for (const section of document.querySelectorAll("main > section")) section.hidden = section.id !== tab
}

function renderItem(item) {
  const container = $("#item")
  container.replaceChildren()
  if (!item) {
    container.textContent = "No items."
    return
  }

  const url = item.target_url || item.url
  const parsedUrl = URL.canParse(url) ? new URL(url) : null
  if (parsedUrl && ["http:", "https:"].includes(parsedUrl.protocol)) {
    const link = document.createElement("a")
    link.href = parsedUrl.href
    link.target = "_blank"
    link.rel = "noreferrer"
    link.textContent = url
    container.append(link)
  } else {
    container.append(url)
  }

  for (const text of [item.anchor || item.title || "", item.snippet || ""]) {
    const paragraph = document.createElement("p")
    paragraph.textContent = text
    container.append(paragraph)
  }

  const ratings = document.createElement("p")
  for (const rating of [1, 2, 3, 4, 5]) {
    const button = document.createElement("button")
    button.dataset.rating = rating
    button.textContent = rating
    ratings.append(button)
  }
  container.append(ratings)
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
  try {
    let item
    if (mode === "teacher") {
      const path = $("#review-path").value
      if (path !== reviewPath) {
        const result = await api("/api/import/review-batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }) })
        reviewBatch = result.results
        reviewPath = path
      }
      item = reviewBatch.shift()
    } else {
      const endpoint = mode === "links" ? "/api/link-candidates?limit=1" : "/api/crawler-candidates?limit=1"
      ;[item] = await api(endpoint)
    }
    current = item || null
    renderItem(item)
    updateReviewCounter()
  } catch (error) { status.textContent = error.message }
}

$("#candidate-mode").onchange = updateReviewCounter

$("#item").onclick = async (event) => {
  const button = event.target.closest("[data-rating]")
  if (!button || !current) return
  const mode = $("#candidate-mode").value
  try {
    const endpoint = mode === "teacher" ? "/api/review-rating" : mode === "links" ? "/api/link-rating" : "/api/rating"
    const body = mode === "teacher" ? { item: current, rating: Number(button.dataset.rating), notes: "" } : mode === "links" ? { link_id: current.id, rating: Number(button.dataset.rating) } : { result_id: current.id, rating: Number(button.dataset.rating) }
    await api(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    $("#load-review").click()
  } catch (error) { status.textContent = error.message }
}

setInterval(async () => { try { const result = await api("/api/pipeline-status"); $("#pipeline-output").textContent = (result.job.output || ["No job started."]).join("\n") } catch {} }, 1200)
