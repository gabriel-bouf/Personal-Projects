
let pollingInterval = null;
let currentCallAmount = 0;
let raiseMin = 0;
let raiseMax = 0;
let currentPot = 0;
let botCardsManualReveal = false;
let resultShown = false;
let lastRoundCount = 0;
let currentGameState = null;

let currentMode = 'pvb';
let bvbPaused = false;

function buyIn() {
    const botType = document.getElementById('bot-type').value;
    const mode = document.getElementById('game-mode').value;
    const bot2Type = document.getElementById('bot2-type') ?
                     document.getElementById('bot2-type').value : 'random';
    botCardsManualReveal = false;
    resultShown = false;
    lastRoundCount = 0;
    currentMode = mode;
    document.getElementById('game-result-overlay').className = 'game-result-overlay';
    document.getElementById('game-result-overlay').style.display = 'none';
    document.getElementById('round-winner-banner').style.display = 'none';

    fetch('/api/buyin', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ bot_type: botType, mode: mode, bot2_type: bot2Type })
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'started' || data.status === 'rebuy') {
            if (data.status === 'started') {
                document.getElementById('logs-list').innerHTML = '';
            }
            startPolling();
            // Show/hide BvB controls
            document.getElementById('bvb-controls').style.display =
                (mode === 'bvb') ? 'flex' : 'none';
            document.getElementById('keyboard-hints').style.display =
                (mode === 'bvb') ? 'none' : 'block';
        } else {
            alert(data.message || 'Could not start game');
        }
    })
    .catch(err => console.error('buyIn fetch error:', err));
}

// Legacy compat
function startGame() { buyIn(); }

function onModeChange() {
    currentMode = document.getElementById('game-mode').value;
    const bot2Select = document.getElementById('bot2-select');
    bot2Select.style.display = (currentMode === 'bvb') ? 'flex' : 'none';
}

function toggleBvbPause() {
    bvbPaused = !bvbPaused;
    const endpoint = bvbPaused ? '/api/bvb/pause' : '/api/bvb/resume';
    document.getElementById('btn-bvb-pause').innerText = bvbPaused ? 'Resume' : 'Pause';
    fetch(endpoint, { method: 'POST', headers: {'Content-Type': 'application/json'} })
        .catch(err => console.error('BvB pause fetch error:', err));
}

function setBvbSpeed(speed, btn) {
    fetch('/api/bvb/speed', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ speed: speed })
    }).catch(err => console.error('BvB speed fetch error:', err));
    // Highlight active speed button
    document.querySelectorAll('.speed-control .btn-preset').forEach(b => {
        b.classList.remove('active-speed');
    });
    if (btn) btn.classList.add('active-speed');
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(updateState, 600);
}

function updateState() {
    fetch('/api/state')
        .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(state => {
            renderGame(state);

            const statusEl = document.getElementById('game-status');
            const startArea = document.getElementById('start-area');

            if (state.waiting_for_buyin) {
                // Human busted - show buy-in button
                statusEl.innerText = "Busted!";
                statusEl.style.background = "#e57373";
                startArea.style.display = "flex";
            } else if (!state.game_running && state.logs && state.logs.length > 0) {
                statusEl.innerText = "Game Over";
                statusEl.style.background = "#e57373";
                startArea.style.display = "flex";
            } else if (state.game_running) {
                statusEl.innerText = "Live";
                statusEl.style.background = "#66bb6a";
                startArea.style.display = "none";
            }

            // Update session info bar
            if (state.persistent_human_stack !== undefined && state.game_running) {
                const sessionInfo = document.getElementById('session-info');
                sessionInfo.style.display = 'flex';
                document.getElementById('persistent-human-stack').innerText = state.persistent_human_stack;
                document.getElementById('persistent-bot-stack').innerText = state.persistent_bot_stack;
                document.getElementById('human-buyin-total').innerText = state.human_total_buyin || 0;
                document.getElementById('bot-buyin-total').innerText = state.bot_total_buyin || 0;
            } else {
                document.getElementById('session-info').style.display = 'none';
            }
        })
        .catch(err => console.error('updateState fetch error:', err));
}

function toggleBotCards() {
    botCardsManualReveal = !botCardsManualReveal;
}

function renderGame(state) {
    currentGameState = state;

    if (state.mode === 'bvb') {
        renderBvbGame(state);
        return;
    }

    // PvB mode — ensure controls visible
    document.getElementById('controls').style.display = '';
    document.getElementById('keyboard-hints').style.display = '';
    document.getElementById('bvb-controls').style.display = 'none';
    const humanNameEl = document.querySelector('.human-seat .name');
    if (humanNameEl) humanNameEl.innerText = 'You';

    // 1. Logs
    const logsDiv = document.getElementById('logs-list');
    logsDiv.innerHTML = state.logs.map(l => {
        // Color-code log entries
        if (l.startsWith('YOU WIN')) return `<div class="log-win">${l}</div>`;
        if (l.startsWith('Bot wins')) return `<div class="log-lose">${l}</div>`;
        if (l.startsWith('---')) return `<div class="log-round">${l}</div>`;
        return `<div>${l}</div>`;
    }).join('');
    logsDiv.scrollTop = logsDiv.scrollHeight;

    // 2. Pot & Stacks
    currentPot = state.pot || 0;
    document.getElementById('pot-amount').innerText = currentPot;

    let humanStack = 1000, botStack = 1000;
    if (state.seats && state.seats.length >= 2) {
        state.seats.forEach(seat => {
            if (seat.name === 'Human') {
                humanStack = seat.stack;
                document.getElementById('human-stack').innerText = seat.stack;
            } else {
                botStack = seat.stack;
                document.getElementById('bot-stack').innerText = seat.stack;
            }
        });
    }

    // 3. Bot name
    if (state.bot_type) {
        document.getElementById('bot-name').innerText = state.bot_type;
    }

    // 4. Street & Round
    const streetEl = document.getElementById('street-display');
    if (state.round_count) {
        let streetText = state.street ? state.street.toUpperCase() : '';
        streetEl.innerText = `ROUND ${state.round_count} ${streetText ? '— ' + streetText : ''}`;
    }

    // 5. Cards
    renderCards('board-cards', state.community_card, 5);
    renderCards('human-cards', state.hole_card, 2);

    // 6. Bot cards - show if showdown or manually revealed
    const shouldReveal = state.showdown || state.bot_cards_revealed || botCardsManualReveal;
    renderBotCards(state.bot_cards, shouldReveal);

    // 7. Bot action bubble
    const bubble = document.getElementById('bot-action-bubble');
    if (state.last_bot_action) {
        bubble.style.display = 'block';
        const parts = state.last_bot_action.split(' ');
        const actionName = parts[0].toUpperCase();
        const amount = parts[1] || '';
        bubble.className = 'bot-action-bubble action-' + parts[0].toLowerCase();
        bubble.innerText = amount ? `${actionName} ${amount}` : actionName;
    } else {
        bubble.style.display = 'none';
    }

    // 8. Round winner banner
    renderRoundWinner(state);

    // 9. Controls
    const controls = document.getElementById('controls');
    const actionInfo = document.getElementById('action-info');
    const raiseControls = document.getElementById('raise-controls');
    const actionButtons = document.getElementById('action-buttons');
    const nextHandArea = document.getElementById('next-hand-area');

    if (state.waiting_for_next_hand) {
        // Round over - show "Next Hand" button
        controls.style.opacity = "1";
        controls.style.pointerEvents = "auto";
        actionInfo.innerText = "Round over — review the hand";
        actionInfo.style.color = "#66bb6a";
        actionButtons.style.display = "none";
        raiseControls.style.display = "none";
        nextHandArea.style.display = "block";

    } else if (state.waiting_for_action) {
        controls.style.opacity = "1";
        controls.style.pointerEvents = "auto";
        actionInfo.innerText = "YOUR TURN";
        actionInfo.style.color = "#ffb74d";
        actionButtons.style.display = "flex";
        nextHandArea.style.display = "none";

        // Use server-computed values (already validated)
        const callAmount = state.call_amount || 0;
        currentCallAmount = callAmount;

        // Parse valid actions
        let canFold = false, canCall = false, canRaise = false;
        if (state.valid_actions) {
            state.valid_actions.forEach(a => {
                if (a.action === 'fold') canFold = true;
                if (a.action === 'call') canCall = true;
                if (a.action === 'raise') canRaise = true;
            });
        }

        // Override canRaise with server validation
        if (state.can_raise === false) canRaise = false;

        // Update buttons
        document.getElementById('btn-fold').disabled = !canFold;

        const callBtn = document.getElementById('btn-call');
        callBtn.disabled = !canCall;
        if (callAmount === 0) {
            callBtn.innerText = 'Check';
            callBtn.className = 'btn btn-check';
        } else if (state.human_stack !== undefined && callAmount >= state.human_stack) {
            callBtn.innerText = `All-In ${callAmount}`;
            callBtn.className = 'btn btn-allin';
        } else {
            callBtn.innerText = `Call ${callAmount}`;
            callBtn.className = 'btn btn-check';
        }

        document.getElementById('btn-raise').disabled = !canRaise;

        // Update raise slider limits — only reset value when limits change
        if (canRaise && state.raise_limits) {
            const newMin = state.raise_limits.min || 0;
            const newMax = state.raise_limits.max || 0;
            const slider = document.getElementById('raise-slider');

            if (newMin !== raiseMin || newMax !== raiseMax) {
                // Limits changed (new action prompt) — reset slider
                raiseMin = newMin;
                raiseMax = newMax;
                slider.min = raiseMin;
                slider.max = raiseMax;
                slider.value = raiseMin;
                document.getElementById('raise-amount-display').innerText = raiseMin;
            }
        } else {
            // Hide raise controls if can't raise
            raiseControls.style.display = 'none';
        }

    } else {
        controls.style.opacity = "0.5";
        controls.style.pointerEvents = "none";
        raiseControls.style.display = "none";
        nextHandArea.style.display = "none";
        actionButtons.style.display = "flex";
        if (state.game_running) {
            actionInfo.innerText = "Opponent thinking...";
        } else {
            actionInfo.innerText = "Game not active";
        }
        actionInfo.style.color = "#b0bec5";
    }
}

function renderRoundWinner(state) {
    const banner = document.getElementById('round-winner-banner');

    if (state.waiting_for_next_hand && state.round_winner) {
        const isHumanWin = state.round_winner === 'you';
        const pot = state.round_winner_amount || 0;

        if (isHumanWin) {
            banner.className = 'round-winner-banner winner-you';
            banner.innerHTML = `You win <span class="win-amount">+${pot}</span>`;
        } else {
            banner.className = 'round-winner-banner winner-bot';
            banner.innerHTML = `Bot wins <span class="win-amount">+${pot}</span>`;
        }
        banner.style.display = 'block';
    } else if (!state.waiting_for_next_hand) {
        banner.style.display = 'none';
    }
}

function renderBotCards(botCards, revealed) {
    const container = document.getElementById('bot-cards');

    if (revealed && botCards && botCards.length > 0) {
        let html = '';
        botCards.forEach(cardStr => {
            if (!cardStr || cardStr.length < 2) return;
            let suit = cardStr[0];
            let rank = cardStr.substring(1);
            let colorClass = (suit === 'H' || suit === 'D') ? 'red' : 'black';
            html += `<div class="card ${colorClass}">${rank}${getSuitSymbol(suit)}</div>`;
        });
        container.innerHTML = html;
    } else {
        container.innerHTML = '<div class="card back"></div><div class="card back"></div>';
    }
}

function getSuitSymbol(s) {
    if (s === 'S') return '\u2660';
    if (s === 'H') return '\u2665';
    if (s === 'D') return '\u2666';
    if (s === 'C') return '\u2663';
    return s;
}

function renderCards(elementId, cardsData, maxPlaceholder) {
    const container = document.getElementById(elementId);
    let html = '';

    if (cardsData && cardsData.length > 0) {
        cardsData.forEach(cardStr => {
            if (!cardStr || cardStr.length < 2) return;
            let suit = cardStr[0];
            let rank = cardStr.substring(1);
            let colorClass = (suit === 'H' || suit === 'D') ? 'red' : 'black';
            html += `<div class="card ${colorClass}">${rank}${getSuitSymbol(suit)}</div>`;
        });
    }

    // Fill placeholders for board cards
    const currentCount = cardsData ? cardsData.length : 0;
    for (let i = 0; i < maxPlaceholder - currentCount; i++) {
        if (elementId === 'board-cards') {
            html += `<div class="card-placeholder"></div>`;
        }
    }

    container.innerHTML = html;
}

function sendAction(actionType, amount) {
    // Hide raise slider after action
    document.getElementById('raise-controls').style.display = 'none';

    fetch('/api/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ action: actionType, amount: amount })
    }).then(() => updateState())
    .catch(err => console.error('sendAction fetch error:', err));
}

function sendCallAction() {
    sendAction('call', currentCallAmount);
}

function sendNextHand() {
    document.getElementById('round-winner-banner').style.display = 'none';
    fetch('/api/next_hand', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    }).then(() => updateState())
    .catch(err => console.error('nextHand fetch error:', err));
}

function toggleRaiseControls() {
    const raiseControls = document.getElementById('raise-controls');
    if (raiseControls.style.display === 'none' || raiseControls.style.display === '') {
        raiseControls.style.display = 'block';
    } else {
        raiseControls.style.display = 'none';
    }
}

function sendRaiseAction() {
    const raiseControls = document.getElementById('raise-controls');
    if (raiseControls.style.display === 'none' || raiseControls.style.display === '') {
        toggleRaiseControls();
    } else {
        // Confirm raise with current slider value
        const amount = parseInt(document.getElementById('raise-slider').value);
        sendAction('raise', amount);
    }
}

function renderBvbGame(state) {
    // 1. Logs
    const logsDiv = document.getElementById('logs-list');
    logsDiv.innerHTML = state.logs.map(l => {
        if (l.startsWith('Winner')) return `<div class="log-win">${l}</div>`;
        if (l.startsWith('---')) return `<div class="log-round">${l}</div>`;
        return `<div>${l}</div>`;
    }).join('');
    logsDiv.scrollTop = logsDiv.scrollHeight;

    // 2. Pot
    document.getElementById('pot-amount').innerText = state.pot || 0;

    // 3. Stacks — bot1 in "human" slot (bottom), bot2 in "bot" slot (top)
    if (state.seats && state.seats.length >= 2) {
        state.seats.forEach((seat, i) => {
            if (i === 0) {
                document.getElementById('human-stack').innerText = seat.stack;
            } else {
                document.getElementById('bot-stack').innerText = seat.stack;
            }
        });
    }

    // 4. Names
    document.getElementById('bot-name').innerText = state.bot2_name || 'Bot 2';
    // For bot1, we reuse the human name area — update it via JS
    const humanNameEl = document.querySelector('.human-seat .name');
    if (humanNameEl) humanNameEl.innerText = state.bot1_name || 'Bot 1';

    // 5. Street & Round
    const streetEl = document.getElementById('street-display');
    if (state.round_count) {
        let streetText = state.street ? state.street.toUpperCase() : '';
        streetEl.innerText = `ROUND ${state.round_count} ${streetText ? '— ' + streetText : ''}`;
    }

    // 6. Community cards
    renderCards('board-cards', state.community_card, 5);

    // 7. Both bots' cards always visible
    renderBotCards(state.bot2_cards || state.bot_cards, true);
    renderCards('human-cards', state.bot1_cards || [], 2);

    // 8. Action bubbles
    const bubble = document.getElementById('bot-action-bubble');
    if (state.bot2_last_action) {
        bubble.style.display = 'block';
        const parts = state.bot2_last_action.split(' ');
        bubble.className = 'bot-action-bubble action-' + parts[0].toLowerCase();
        bubble.innerText = parts[0].toUpperCase() + (parts[1] ? ' ' + parts[1] : '');
    } else {
        bubble.style.display = 'none';
    }

    // 9. Hide PvB controls, show BvB controls
    document.getElementById('controls').style.display = 'none';
    document.getElementById('keyboard-hints').style.display = 'none';
    document.getElementById('bvb-controls').style.display = 'flex';
}

function calcLogRaise(level) {
    // level: 1 = min raise, 9 = all-in
    // Formula: amount = raiseMin * (raiseMax/raiseMin)^((level-1)/8)
    if (raiseMin <= 0 || raiseMax <= 0 || raiseMin >= raiseMax) {
        return raiseMax;
    }
    if (level === 1) return raiseMin;
    if (level === 9) return raiseMax;
    const ratio = raiseMax / raiseMin;
    const exponent = (level - 1) / 8;
    const amount = Math.round(raiseMin * Math.pow(ratio, exponent));
    return Math.max(raiseMin, Math.min(raiseMax, amount));
}

function setRaise(preset) {
    const slider = document.getElementById('raise-slider');
    let value;
    if (preset === 'min') value = raiseMin;
    else if (preset === 'half') value = Math.max(raiseMin, Math.floor(currentPot / 2));
    else if (preset === 'pot') value = Math.max(raiseMin, Math.min(currentPot, raiseMax));
    else if (preset === 'allin') value = raiseMax;
    else value = raiseMin;

    value = Math.max(raiseMin, Math.min(raiseMax, value));
    slider.value = value;
    document.getElementById('raise-amount-display').innerText = value;
}

function showGameResult(state) {
    if (resultShown || !state.game_result) return;
    resultShown = true;

    const overlay = document.getElementById('game-result-overlay');
    const text = document.getElementById('game-result-text');
    const sub = document.getElementById('game-result-sub');

    let humanStack = 0, botStack = 0;
    if (state.seats) {
        state.seats.forEach(s => {
            if (s.name === 'Human') humanStack = s.stack;
            else botStack = s.stack;
        });
    }

    if (state.game_result === 'win') {
        text.innerText = 'VICTORY';
        sub.innerText = `${humanStack} vs ${botStack} — click to dismiss`;
        overlay.className = 'game-result-overlay show win';
    } else if (state.game_result === 'lose') {
        text.innerText = 'DEFEAT';
        sub.innerText = `${humanStack} vs ${botStack} — click to dismiss`;
        overlay.className = 'game-result-overlay show lose';
    } else {
        text.innerText = 'DRAW';
        sub.innerText = `${humanStack} vs ${botStack} — click to dismiss`;
        overlay.className = 'game-result-overlay show draw';
    }
}

// Slider live update + keyboard shortcuts
document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('raise-slider');
    slider.addEventListener('input', () => {
        document.getElementById('raise-amount-display').innerText = slider.value;
    });

    document.addEventListener('keydown', (e) => {
        // Don't capture keys when typing in an input/select
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
        if (!currentGameState) return;

        const key = e.key.toLowerCase();

        // Action shortcuts (only when it's your turn)
        if (currentGameState.waiting_for_action && currentGameState.mode !== 'bvb') {
            if (key === 'f') {
                e.preventDefault();
                sendAction('fold', 0);
                return;
            }
            if (key === 'c') {
                e.preventDefault();
                sendCallAction();
                return;
            }
            if (key === 'r') {
                e.preventDefault();
                toggleRaiseControls();
                return;
            }
            if (key === 'escape') {
                e.preventDefault();
                document.getElementById('raise-controls').style.display = 'none';
                return;
            }
            // Logarithmic raise: keys 1-9
            if (key >= '1' && key <= '9' && currentGameState.can_raise) {
                e.preventDefault();
                const level = parseInt(key);
                const amount = calcLogRaise(level);
                sendAction('raise', amount);
                return;
            }
        }

        // Next Hand shortcut
        if (currentGameState.waiting_for_next_hand) {
            if (key === ' ' || key === 'enter') {
                e.preventDefault();
                sendNextHand();
                return;
            }
        }
    });
});
