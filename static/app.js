const state = {
  results: [],
  currentIndex: 0,
  undoStack: [],
  mode: "pages",
}

const items = document.querySelector("#items")
const statusBox = document.querySelector("#status")
const statsBox = document.querySelector("#stats")
const queryInput = document.querySelector("#search-query")
const searchButton = document.querySelector("#search")
const candidateModeInput = document.querySelector("#candidate-mode")
const crawlerCsvPathInput = document.querySelector("#crawler-csv-path")
const crawlerLimitInput = document.querySelector("#crawler-limit")
const crawlerUnlabeledOnlyInput = document.querySelector("#crawler-unlabeled-only")
const importCrawlerButton = document.querySelector("#import-crawler")
const loadCrawlerButton = document.querySelector("#load-crawler")
const pipelineState = document.querySelector("#pipeline-state")
const pipelineData = document.querySelector("#pipeline-data")
const pipelineOutput = document.querySelector("#pipeline-output")
const pipelineLimit = document.querySelector("#pipeline-limit")
const pipelineWorkers = document.querySelector("#pipeline-workers")
const pipelineModel = document.querySelector("#pipeline-model")
const pipelineCrawlDb = document.querySelector("#pipeline-crawl-db")
const pipelineLog = document.querySelector("#pipeline-log")
const manualHint = document.querySelector("#manual-hint")

const ratings = [
  { value: 1, label: "Reject" },
  { value: 2, label: "Bad" },
  { value: 3, label: "Unsure" },
  { value: 4, label: "Good" },
  { value: 5, label: "Great" },
]

const linkRatings = [
  { value: 1, label: "Discard" },
  { value: 2, label: "Weak" },
  { value: 3, label: "Unsure" },
  { value: 4, label: "Follow" },
  { value: 5, label: "Strong" },
]

const defaultPaths = {
  pages: "data/pageverdict_candidates.jsonl",
  links: "data/link_candidates.jsonl",
  teacher: "data/review_batches/batch-0001.jsonl",
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
}

function setStatus(message, isError = false) {
  statusBox.textContent = message || ""
  statusBox.classList.toggle("is-error", isError)
}

async function fetchJson(url, options) {
  const response = await fetch(url, options)
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const payload = await response.json()
      detail = payload.detail || detail
    } catch {
      // Keep the HTTP status when the response is not JSON.
    }
    throw new Error(detail)
  }
  return response.json()
}

function renderPipeline(status) {
  const job = status.job || {}
  const data = status.data || {}
  pipelineState.textContent = job.running ? "running" : "idle"
  pipelineState.classList.toggle("running", Boolean(job.running))
  if (job.running) pipelineLog.open = true
  pipelineData.innerHTML = [
    ["SERP", data.serp_rows], ["snapshots", data.snapshots],
    ["teacher labels", data.teacher_labels], ["reviewed", data.reviewed_labels],
    ["page model", data.page_model ? "ready" : "missing"],
    ["link model", data.link_model ? "ready" : "missing"],
  ].map(([label, value]) => `<span class="stat"><strong>${escapeHtml(value)}</strong> ${escapeHtml(label)}</span>`).join("")
  pipelineOutput.textContent = (job.output || ["No job started."]).join("\n")
  pipelineOutput.scrollTop = pipelineOutput.scrollHeight
}

async function loadPipelineStatus() {
  try {
    renderPipeline(await fetchJson("/api/pipeline-status"))
  } catch (error) {
    pipelineOutput.textContent = `Pipeline status failed: ${error.message}`
  }
}

async function runPipeline(action) {
  const payload = {
    action,
    limit: Number(pipelineLimit.value) || undefined,
    workers: Number(pipelineWorkers.value) || undefined,
    model: pipelineModel.value.trim() || undefined,
    crawl_db: pipelineCrawlDb.value.trim() || undefined,
  }
  try {
    renderPipeline(await fetchJson("/api/pipeline-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }))
  } catch (error) {
    pipelineOutput.textContent = `Pipeline start failed: ${error.message}`
  }
}

function ratingCounts(counts) {
  return [1, 2, 3, 4, 5]
    .map((rating) => `${rating}:${counts[String(rating)] || 0}`)
    .join(" ")
}

async function loadStats() {
  const endpoint = state.mode === "links" ? "/api/link-stats" : "/api/stats"
  const stats = await fetchJson(endpoint)
  const noun = state.mode === "links" ? "links" : state.mode === "teacher" ? "page labels" : "pages"
  statsBox.innerHTML = [
    `<span class="stat"><strong>${stats.results}</strong> ${noun}</span>`,
    `<span class="stat"><strong>${stats.rated}</strong> rated</span>`,
    `<span class="stat">ratings <strong>${ratingCounts(stats.ratings || {})}</strong></span>`,
  ].join("")
}

function currentResult() {
  return state.results[state.currentIndex] || null
}

function labelText(result) {
  if (!result || result.rating === null || result.rating === undefined) return "unlabeled"
  return `rated ${result.rating} (${result.label})`
}

function confidenceText(result) {
  if (result?.pageverdict_score === null || result?.pageverdict_score === undefined) return ""
  const score = Number(result.pageverdict_score)
  const distance = Math.abs(score - 0.5)
  return `pv ${score.toFixed(3)} · conf ${distance.toFixed(3)}`
}

function crawlerMeta(result) {
  if (result?.pageverdict_score === null || result?.pageverdict_score === undefined) return ""
  return [
    result.pageverdict_decision ? `<span class="pill">${escapeHtml(result.pageverdict_decision)}</span>` : "",
    confidenceText(result) ? `<span class="pill">${escapeHtml(confidenceText(result))}</span>` : "",
    result.crawler_source_table ? `<span class="pill">${escapeHtml(result.crawler_source_table)}</span>` : "",
    result.crawler_exclusion_reason ? `<span class="pill">${escapeHtml(result.crawler_exclusion_reason)}</span>` : "",
  ].join("")
}

function renderRatingButtons(result) {
  const labels = state.mode === "links" ? linkRatings : ratings
  return labels
    .map((rating) => {
      const active = result?.rating === rating.value ? " active" : ""
      return `
        <button class="rating rating-${rating.value}${active}" data-rating="${rating.value}" type="button">
          <kbd>${rating.value}</kbd>
          <span>${rating.label}</span>
        </button>
      `
    })
    .join("")
}

function scorePill(label, value, digits = 3) {
  if (value === null || value === undefined || value === "") return ""
  const number = Number(value)
  const text = Number.isFinite(number) ? number.toFixed(digits) : String(value)
  return `<span class="pill">${escapeHtml(label)} ${escapeHtml(text)}</span>`
}

function boolText(value) {
  return value ? "yes" : "no"
}

function renderLinkCurrent(result) {
  const targetUrl = result.target_url || ""
  const parentUrl = result.parent_url || ""
  items.innerHTML = `
    <article class="deck-card">
      <div class="deck-progress">
        <span>${state.currentIndex + 1} / ${state.results.length}</span>
        <span class="${result.rating ? "rated" : "unrated"}">${escapeHtml(labelText(result))}</span>
      </div>
      <div class="card-head">
        <div class="title-row">
          <h2>${escapeHtml(result.anchor || targetUrl)}</h2>
          <a class="pill" href="${escapeHtml(targetUrl)}" target="_blank" rel="noreferrer">Open target</a>
        </div>
        <div class="url">${escapeHtml(targetUrl)}</div>
        <div class="meta">
          <span class="pill">link</span>
          <span class="pill">${escapeHtml(result.source)}</span>
          <span class="pill">selected ${escapeHtml(boolText(result.selected))}</span>
          <span class="pill">enqueue ${escapeHtml(boolText(result.should_enqueue))}</span>
          ${result.rejection_reason ? `<span class="pill">${escapeHtml(result.rejection_reason)}</span>` : ""}
          ${scorePill("raw", result.raw_score)}
          ${scorePill("lv", result.linkverdict_score)}
          ${result.linkverdict_label ? `<span class="pill">lv ${escapeHtml(result.linkverdict_label)}</span>` : ""}
          ${scorePill("parent pv", result.parent_pageverdict_score)}
          ${scorePill("target pv", result.target_pageverdict_score)}
          ${result.target_pageverdict_decision ? `<span class="pill">${escapeHtml(result.target_pageverdict_decision)}</span>` : ""}
          ${result.target_status ? `<span class="pill">${escapeHtml(result.target_status)}</span>` : ""}
        </div>
      </div>
      <div class="link-grid">
        <div>
          <div class="field-label">Parent</div>
          <a href="${escapeHtml(parentUrl)}" target="_blank" rel="noreferrer">${escapeHtml(parentUrl || "No parent URL")}</a>
        </div>
        <div>
          <div class="field-label">Target host</div>
          <span>${escapeHtml(result.target_host || "")}</span>
        </div>
        <div>
          <div class="field-label">Parent host</div>
          <span>${escapeHtml(result.parent_host || "")}</span>
        </div>
        <div>
          <div class="field-label">Target outcome</div>
          <span>${escapeHtml(result.target_exclusion_reason || result.target_pageverdict_label || "unknown")}</span>
        </div>
      </div>
      ${result.notes ? `<p class="notes">${escapeHtml(result.notes)}</p>` : ""}
      <div class="rating-grid">
        ${renderRatingButtons(result)}
      </div>
      <div class="secondary-actions">
        <button data-action="prev" type="button">Prev</button>
        <button data-action="next" type="button">Next</button>
        ${state.mode === "teacher" ? "" : '<button data-action="undo" type="button">Undo</button>'}
      </div>
    </article>
  `
}

function renderCurrent() {
  const result = currentResult()
  if (!result) {
    items.innerHTML = `<div class="empty">No results loaded.</div>`
    return
  }

  if (state.mode === "links") {
    renderLinkCurrent(result)
    return
  }

  const displayUrl = result.display_url || result.url
  items.innerHTML = `
    <article class="deck-card">
      <div class="deck-progress">
        <span>${state.currentIndex + 1} / ${state.results.length}</span>
        <span class="${result.rating ? "rated" : "unrated"}">${escapeHtml(labelText(result))}</span>
      </div>
      <div class="card-head">
        <div class="title-row">
          <h2>${escapeHtml(result.title || displayUrl)}</h2>
          <a class="pill" href="${escapeHtml(result.url)}" target="_blank" rel="noreferrer">Open</a>
        </div>
        <div class="url">${escapeHtml(displayUrl)}</div>
        <div class="meta">
          ${result.query ? `<span class="pill">query ${escapeHtml(result.query)}</span>` : ""}
          ${result.page_number ? `<span class="pill">page ${escapeHtml(result.page_number)}</span>` : ""}
          ${result.rank ? `<span class="pill">rank ${escapeHtml(result.rank)}</span>` : ""}
          <span class="pill">${escapeHtml(result.source)}</span>
          ${result.teacher ? `<span class="pill">teacher ${escapeHtml(result.teacher)}</span>` : ""}
          ${result.model ? `<span class="pill">${escapeHtml(result.model)}</span>` : ""}
          ${crawlerMeta(result)}
        </div>
      </div>
      <p class="snippet">${escapeHtml(result.snippet || "No snippet returned.")}</p>
      ${result.notes ? `<p class="notes">${escapeHtml(result.notes)}</p>` : ""}
      <div class="rating-grid">
        ${renderRatingButtons(result)}
      </div>
      <div class="secondary-actions">
        <button data-action="prev" type="button">Prev</button>
        <button data-action="next" type="button">Next</button>
        ${state.mode === "teacher" ? "" : '<button data-action="undo" type="button">Undo</button>'}
      </div>
    </article>
  `
}

async function search() {
  const query = queryInput.value.trim()
  if (!query) return

  searchButton.disabled = true
  setStatus("Searching Serper pages 1-4...")
  items.innerHTML = `<div class="empty">Loading SERP results...</div>`

  try {
    const payload = await fetchJson("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    })
    state.mode = "pages"
    state.results = payload.results || []
    state.currentIndex = 0
    state.undoStack = []
    renderCurrent()
    await loadStats()
    setStatus(`Loaded ${state.results.length} results from Serper pages 1-4.`)
    queryInput.blur()
  } catch (error) {
    setStatus(`Search failed: ${error.message}`, true)
    items.innerHTML = ""
  } finally {
    searchButton.disabled = false
  }
}

function crawlerLimit() {
  const value = Number(crawlerLimitInput.value || 200)
  return Math.max(1, Math.min(value, 2000))
}

function crawlerUnlabeledOnly() {
  return Boolean(crawlerUnlabeledOnlyInput.checked)
}

function currentMode() {
  return candidateModeInput.value
}

function syncMode() {
  const mode = currentMode()
  state.mode = mode
  if (!crawlerCsvPathInput.value.trim() || Object.values(defaultPaths).includes(crawlerCsvPathInput.value.trim())) {
    crawlerCsvPathInput.value = defaultPaths[mode]
  }
  manualHint.textContent = {
    pages: "Import crawler page candidates, then rate one page at a time.",
    links: "Import crawler link candidates, then decide which links are worth following.",
    teacher: "Load a review batch to correct the local teacher before training.",
  }[mode]
}

async function importCrawlerCandidates() {
  syncMode()
  const mode = currentMode()
  const path = crawlerCsvPathInput.value.trim() || defaultPaths[mode]
  const endpoint = mode === "links"
    ? "/api/import/link-candidates"
    : mode === "teacher"
      ? "/api/import/review-batch"
      : "/api/import/crawler-pageverdict"
  importCrawlerButton.disabled = true
  loadCrawlerButton.disabled = true
  setStatus("Importing candidate JSONL...")
  items.innerHTML = `<div class="empty">Loading candidates...</div>`

  try {
    const payload = await fetchJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mode === "teacher" ? { path } : {
        path,
        limit: crawlerLimit(),
        unlabeled_only: crawlerUnlabeledOnly(),
      }),
    })
    state.mode = mode
    state.results = payload.results || []
    state.currentIndex = 0
    state.undoStack = []
    renderCurrent()
    await loadStats()
    setStatus(`Imported ${payload.rows_read} ${mode} rows. Loaded ${state.results.length} candidates.`)
  } catch (error) {
    setStatus(`Import failed: ${error.message}`, true)
    items.innerHTML = ""
  } finally {
    importCrawlerButton.disabled = false
    loadCrawlerButton.disabled = false
  }
}

async function loadCrawlerCandidates() {
  syncMode()
  const mode = currentMode()
  if (mode === "teacher") {
    setStatus("Choose a teacher-review batch and click Import.")
    return
  }
  const endpoint = mode === "links" ? "/api/link-candidates" : "/api/crawler-candidates"
  loadCrawlerButton.disabled = true
  setStatus("Loading candidates by lowest confidence...")
  items.innerHTML = `<div class="empty">Loading candidates...</div>`

  try {
    const params = new URLSearchParams({
      limit: String(crawlerLimit()),
      unlabeled_only: String(crawlerUnlabeledOnly()),
    })
    state.mode = mode
    state.results = await fetchJson(`${endpoint}?${params}`)
    state.currentIndex = 0
    state.undoStack = []
    renderCurrent()
    await loadStats()
    setStatus(`Loaded ${state.results.length} candidates.`)
  } catch (error) {
    setStatus(`Load failed: ${error.message}`, true)
    items.innerHTML = ""
  } finally {
    loadCrawlerButton.disabled = false
  }
}

function move(delta) {
  if (state.results.length === 0) return
  state.currentIndex =
    (state.currentIndex + delta + state.results.length) % state.results.length
  renderCurrent()
}

async function persistRating(result, rating, notes = "") {
  const endpoint = state.mode === "links" ? "/api/link-rating" : state.mode === "teacher" ? "/api/review-rating" : "/api/rating"
  const body = state.mode === "links"
    ? { link_id: result.id, rating, notes }
    : state.mode === "teacher"
      ? { item: result, rating, notes }
    : { result_id: result.id, rating, notes }
  await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

function mappedLabel(rating) {
  if (rating === null || rating === undefined) return null
  if (rating <= 2) return "negative"
  if (rating === 3) return "skip"
  return "positive"
}

async function rateCurrent(rating) {
  const result = currentResult()
  if (!result) return

  const previous = {
    index: state.currentIndex,
    id: result.id,
    rating: result.rating ?? null,
    label: result.label ?? null,
    notes: result.notes || "",
  }

  await persistRating(result, rating, result.notes || "")
  result.rating = rating
  result.label = mappedLabel(rating)
  state.undoStack.push(previous)

  if (state.currentIndex < state.results.length - 1) {
    state.currentIndex += 1
  }
  renderCurrent()
  await loadStats()
}

async function undoLast() {
  const previous = state.undoStack.pop()
  if (!previous) return

  const index = state.results.findIndex((result) => result.id === previous.id)
  if (index < 0) return

  const result = state.results[index]
  await persistRating(result, previous.rating, previous.notes)
  result.rating = previous.rating
  result.label = previous.label
  result.notes = previous.notes
  state.currentIndex = index
  renderCurrent()
  await loadStats()
}

items.addEventListener("click", async (event) => {
  const button = event.target.closest("button")
  if (!button) return

  button.disabled = true
  setStatus("")
  try {
    if (button.dataset.rating) {
      await rateCurrent(Number(button.dataset.rating))
    } else if (button.dataset.action === "next") {
      move(1)
    } else if (button.dataset.action === "prev") {
      move(-1)
    } else if (button.dataset.action === "undo") {
      await undoLast()
    }
  } catch (error) {
    setStatus(`Action failed: ${error.message}`, true)
  } finally {
    button.disabled = false
  }
})

document.addEventListener("keydown", async (event) => {
  const tag = document.activeElement?.tagName?.toLowerCase()
  if (["input", "select", "textarea"].includes(tag)) return

  if (/^[1-5]$/.test(event.key)) {
    event.preventDefault()
    try {
      await rateCurrent(Number(event.key))
    } catch (error) {
      setStatus(`Action failed: ${error.message}`, true)
    }
  } else if (event.key === "ArrowRight") {
    event.preventDefault()
    move(1)
  } else if (event.key === "ArrowLeft") {
    event.preventDefault()
    move(-1)
  } else if (event.key === "Backspace" || event.key.toLowerCase() === "u") {
    event.preventDefault()
    try {
      await undoLast()
    } catch (error) {
      setStatus(`Undo failed: ${error.message}`, true)
    }
  }
})

searchButton.addEventListener("click", search)
importCrawlerButton.addEventListener("click", importCrawlerCandidates)
loadCrawlerButton.addEventListener("click", loadCrawlerCandidates)
candidateModeInput.addEventListener("change", () => {
  syncMode()
  state.results = []
  state.currentIndex = 0
  state.undoStack = []
  renderCurrent()
  loadStats().catch((error) => setStatus(`Could not load stats: ${error.message}`, true))
})
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault()
    search()
  }
})
crawlerCsvPathInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault()
    importCrawlerCandidates()
  }
})

syncMode()
loadStats().catch((error) => setStatus(`Could not load stats: ${error.message}`, true))
renderCurrent()
loadPipelineStatus()
setInterval(loadPipelineStatus, 1200)

document.querySelectorAll("[data-pipeline]").forEach((button) => {
  button.addEventListener("click", () => runPipeline(button.dataset.pipeline))
})

document.querySelectorAll("[data-open-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    candidateModeInput.value = button.dataset.openMode
    syncMode()
    document.querySelector("#manual-labeling").scrollIntoView({ behavior: "smooth", block: "start" })
  })
})
