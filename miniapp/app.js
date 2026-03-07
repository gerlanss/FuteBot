const tg = window.Telegram?.WebApp;
if (tg) {
  tg.ready();
  tg.expand();
}

const state = {
  chatId: "",
  leagues: [],
  markets: [],
  latestPredictions: [],
  preferences: {
    alerts_enabled: true,
    min_ev: 3,
    favorite_leagues: [],
    bankroll_initial: 1000,
    stake_unit: 1,
  },
};

const elStats = document.getElementById("stats-grid");
const elLeagues = document.getElementById("league-list");
const elLatest = document.getElementById("latest-list");
const elSave = document.getElementById("save-btn");
const elAlerts = document.getElementById("alerts-enabled");
const elMinEv = document.getElementById("min-ev");
const elBankrollInitial = document.getElementById("bankroll-initial");
const elStakeUnit = document.getElementById("stake-unit");
const elBankrollGrid = document.getElementById("bankroll-grid");
const elFilterLeague = document.getElementById("filter-league");
const elFilterMarket = document.getElementById("filter-market");
const elFilterStatus = document.getElementById("filter-status");
const elStatus = document.getElementById("status-line");

function resolveChatId() {
  const qs = new URLSearchParams(window.location.search);
  const fromQuery = qs.get("chat_id");
  const fromTelegram = tg?.initDataUnsafe?.user?.id;
  return String(fromTelegram || fromQuery || "");
}

function fmtNumber(value) {
  if (value === null || value === undefined) return "-";
  return new Intl.NumberFormat("pt-BR").format(value);
}

function fmtMoney(value) {
  if (value === null || value === undefined) return "-";
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
  }).format(value);
}

function renderStats(payload) {
  const cards = [
    ["Fixtures", fmtNumber(payload.summary.fixtures)],
    ["Com stats", fmtNumber(payload.summary.fixtures_com_stats)],
    ["Predicoes", fmtNumber(payload.summary.predictions)],
    ["ROI", `${payload.metrics.roi ?? 0}%`],
  ];

  elStats.innerHTML = cards.map(([label, value]) => `
    <article class="stat-card">
      <span class="label">${label}</span>
      <span class="value">${value}</span>
    </article>
  `).join("");
}

function renderBankroll(bankroll) {
  const cards = [
    ["Banca inicial", fmtMoney(bankroll.initial), bankroll.initial >= 0],
    ["Banca atual", fmtMoney(bankroll.current), bankroll.current >= bankroll.initial],
    ["Lucro", fmtMoney(bankroll.profit_value), bankroll.profit_value >= 0],
    ["Lucro em u", `${bankroll.profit_units > 0 ? "+" : ""}${bankroll.profit_units}u`, bankroll.profit_units >= 0],
  ];

  elBankrollGrid.innerHTML = cards.map(([label, value, isPositive]) => `
    <article class="bankroll-card">
      <span class="label">${label}</span>
      <span class="value ${isPositive ? "positive" : "negative"}">${value}</span>
    </article>
  `).join("");
}

function renderLeagues() {
  elLeagues.innerHTML = state.leagues.map((league) => {
    const checked = state.preferences.favorite_leagues.includes(league.id) ? "checked" : "";
    return `
      <label class="league-item">
        <input type="checkbox" value="${league.id}" ${checked}>
        <span>${league.name}</span>
      </label>
    `;
  }).join("");
}

function renderFilters() {
  const leagueOptions = ['<option value="all">Todas</option>'].concat(
    state.leagues.map((league) => `<option value="${league.id}">${league.name}</option>`)
  );
  elFilterLeague.innerHTML = leagueOptions.join("");

  const marketOptions = ['<option value="all">Todos</option>'].concat(
    state.markets.map((market) => `<option value="${market}">${market}</option>`)
  );
  elFilterMarket.innerHTML = marketOptions.join("");
}

function statusLabel(status) {
  if (status === "win") return "Green";
  if (status === "loss") return "Red";
  return "Aberta";
}

function filterPredictions() {
  const leagueFilter = elFilterLeague.value;
  const marketFilter = elFilterMarket.value;
  const statusFilter = elFilterStatus.value;

  return state.latestPredictions.filter((item) => {
    const matchesLeague = leagueFilter === "all" || String(item.league_id) === leagueFilter;
    const matchesMarket = marketFilter === "all" || item.mercado === marketFilter;
    const matchesStatus = statusFilter === "all" || item.status === statusFilter;
    return matchesLeague && matchesMarket && matchesStatus;
  });
}

function renderLatest(items) {
  if (!items.length) {
    elLatest.innerHTML = '<div class="tip-item">Nenhuma tip encontrada com esse filtro.</div>';
    return;
  }

  elLatest.innerHTML = items.map((item) => {
    const ev = item.ev_percent !== null && item.ev_percent !== undefined
      ? `EV ${item.ev_percent > 0 ? "+" : ""}${item.ev_percent}%`
      : "EV n/d";
    const odd = item.odd ? `Odd ${Number(item.odd).toFixed(2)}` : "Odd n/d";
    const pnl = item.lucro !== null && item.lucro !== undefined
      ? `PnL ${item.lucro > 0 ? "+" : ""}${Number(item.lucro).toFixed(2)}u`
      : "PnL n/d";
    return `
      <article class="tip-item">
        <div class="topline">
          <div>
            <div class="match">${item.home_name} vs ${item.away_name}</div>
            <div class="league">${item.league_name || "Liga"} | ${item.date?.slice(0, 16).replace("T", " ") || ""}</div>
          </div>
          <span class="pill ${item.status}">${statusLabel(item.status)}</span>
        </div>
        <div class="meta">${item.mercado || "mercado"} | ${odd} | ${ev}</div>
        <div class="meta">${item.bookmaker || "referencia"} | ${pnl}</div>
      </article>
    `;
  }).join("");
}

function rerenderHistory() {
  renderLatest(filterPredictions());
}

async function loadState() {
  state.chatId = resolveChatId();
  const url = `/miniapp/api/state?chat_id=${encodeURIComponent(state.chatId)}`;
  const response = await fetch(url);
  const payload = await response.json();

  state.leagues = payload.leagues || [];
  state.markets = payload.markets || [];
  state.latestPredictions = payload.latest_predictions || [];
  state.preferences = payload.preferences || state.preferences;

  elAlerts.checked = !!state.preferences.alerts_enabled;
  elMinEv.value = state.preferences.min_ev ?? 3;
  elBankrollInitial.value = state.preferences.bankroll_initial ?? 1000;
  elStakeUnit.value = state.preferences.stake_unit ?? 1;

  renderStats(payload);
  renderBankroll(payload.bankroll);
  renderLeagues();
  renderFilters();
  rerenderHistory();

  elStatus.textContent = state.chatId
    ? `Conectado ao chat ${state.chatId}.`
    : "Sem chat identificado. Abra pelo menu do bot dentro do Telegram.";
}

async function savePreferences() {
  const selectedLeagues = [...elLeagues.querySelectorAll("input:checked")].map((input) => Number(input.value));
  const payload = {
    chat_id: state.chatId,
    alerts_enabled: elAlerts.checked,
    min_ev: Number(elMinEv.value || 0),
    favorite_leagues: selectedLeagues,
    bankroll_initial: Number(elBankrollInitial.value || 0),
    stake_unit: Number(elStakeUnit.value || 0),
  };

  const response = await fetch("/miniapp/api/preferences", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    elStatus.textContent = "Nao foi possivel salvar agora.";
    return;
  }

  elStatus.textContent = "Preferencias e banca salvas.";
  if (tg?.HapticFeedback) {
    tg.HapticFeedback.notificationOccurred("success");
  }
  await loadState();
}

elSave.addEventListener("click", savePreferences);
elFilterLeague.addEventListener("change", rerenderHistory);
elFilterMarket.addEventListener("change", rerenderHistory);
elFilterStatus.addEventListener("change", rerenderHistory);

loadState().catch(() => {
  elStatus.textContent = "Erro ao carregar a Mini App.";
});
