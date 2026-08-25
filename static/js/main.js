const AIRI_THEME_STORAGE_KEY = 'airi-theme';

function statusToBadge(status) {
    switch (status) {
    case 'known':
        return 'success';
    case 'unknown':
        return 'danger';
    default:
        return 'secondary';
    }
}

function getStoredTheme() {
    try {
        const theme = window.localStorage.getItem(AIRI_THEME_STORAGE_KEY);
        return theme === 'light' || theme === 'dark' ? theme : null;
    } catch (error) {
        return null;
    }
}

function applyTheme(theme, { persist = false } = {}) {
    const selectedTheme = theme === 'dark' ? 'dark' : 'light';
    const root = document.documentElement;
    const toggle = document.getElementById('themeToggle');
    const label = document.getElementById('themeToggleLabel');
    const icon = toggle?.querySelector('i');
    const isDark = selectedTheme === 'dark';

    root.dataset.theme = selectedTheme;
    if (toggle) {
        toggle.setAttribute('aria-pressed', String(isDark));
        toggle.setAttribute('aria-label', isDark ? "Tungi mavzu" : "Yorug' mavzu");
    }
    if (label) {
        label.textContent = isDark ? 'Tungi mavzu' : "Yorug' mavzu";
    }
    if (icon) {
        icon.className = isDark ? 'bi bi-moon-stars' : 'bi bi-sun';
    }
    if (persist) {
        try {
            window.localStorage.setItem(AIRI_THEME_STORAGE_KEY, selectedTheme);
        } catch (error) {
            // Theme selection remains available when browser storage is unavailable.
        }
    }
    document.dispatchEvent(new CustomEvent('airi:themechange', {
        detail: { theme: selectedTheme },
    }));
}

function initialiseTheme() {
    const preferredTheme = window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    applyTheme(getStoredTheme() || preferredTheme);

    document.getElementById('themeToggle')?.addEventListener('click', () => {
        const currentTheme = document.documentElement.dataset.theme;
        applyTheme(currentTheme === 'dark' ? 'light' : 'dark', { persist: true });
    });
}

function initialiseEvidenceModal() {
    const modalElement = document.getElementById('evidenceModal');
    const image = document.getElementById('evidenceModalImage');
    const titleElement = document.getElementById('evidenceModalTitle');
    const subtitleElement = document.getElementById('evidenceModalSubtitle');
    const emptyElement = document.getElementById('evidenceModalEmpty');

    window.AiriUI = window.AiriUI || {};
    window.AiriUI.openEvidence = ({ src, title, subtitle } = {}) => {
        if (!modalElement || !image || !titleElement || !subtitleElement || !emptyElement) {
            return;
        }

        const safeSource = typeof src === 'string' && src.trim() ? src.trim() : '';
        const safeTitle = typeof title === 'string' && title.trim() ? title.trim() : 'Dalil';
        const safeSubtitle = typeof subtitle === 'string' ? subtitle.trim() : '';

        titleElement.textContent = safeTitle;
        subtitleElement.textContent = safeSubtitle;
        subtitleElement.hidden = !safeSubtitle;
        image.alt = safeTitle;
        image.hidden = !safeSource;
        emptyElement.hidden = Boolean(safeSource);
        if (safeSource) {
            image.src = safeSource;
        } else {
            image.removeAttribute('src');
        }

        if (window.bootstrap?.Modal) {
            window.bootstrap.Modal.getOrCreateInstance(modalElement).show();
        }
    };
}

function initialiseEvidenceControls() {
    document.addEventListener('click', (event) => {
        const control = event.target.closest('[data-airi-evidence][data-evidence-src]');
        if (!control) {
            return;
        }
        window.AiriUI?.openEvidence({
            src: control.dataset.evidenceSrc,
            title: control.dataset.evidenceTitle,
            subtitle: control.dataset.evidenceSubtitle,
        });
    });
}

function formatLocalDate(value) {
    const year = value.getFullYear();
    const month = String(value.getMonth() + 1).padStart(2, '0');
    const day = String(value.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function oneCalendarMonthEarlier(value) {
    const targetMonth = value.getMonth() - 1;
    const targetYear = value.getFullYear() + Math.floor(targetMonth / 12);
    const normalizedMonth = (targetMonth + 12) % 12;
    const lastDay = new Date(targetYear, normalizedMonth + 1, 0).getDate();
    return new Date(targetYear, normalizedMonth, Math.min(value.getDate(), lastDay));
}

function initialiseAttendanceFilters() {
    const form = document.querySelector('[data-attendance-filters]');
    const startInput = form?.querySelector('#start_date');
    const endInput = form?.querySelector('#end_date');
    if (!form || !startInput || !endInput) {
        return;
    }

    form.addEventListener('click', (event) => {
        const shortcut = event.target.closest('[data-attendance-shortcut]');
        if (!shortcut) {
            return;
        }
        const end = new Date();
        end.setHours(0, 0, 0, 0);
        const start = new Date(end);
        if (shortcut.dataset.attendanceShortcut === '15-days') {
            start.setDate(start.getDate() - 14);
        } else if (shortcut.dataset.attendanceShortcut === 'one-month') {
            start.setTime(oneCalendarMonthEarlier(end).getTime());
        } else {
            return;
        }
        startInput.value = formatLocalDate(start);
        endInput.value = formatLocalDate(end);
        form.requestSubmit();
    });
}

function initialiseMobileNavigation() {
    const collapseElement = document.getElementById('airiPrimaryNavigation');
    if (!collapseElement) {
        return;
    }

    collapseElement.querySelectorAll('.nav-link, .app-nav__login').forEach((link) => {
        link.addEventListener('click', () => {
            if (collapseElement.classList.contains('show') && window.bootstrap?.Collapse) {
                window.bootstrap.Collapse.getOrCreateInstance(collapseElement).hide();
            }
        });
    });
}

function initialiseDismissControls() {
    document.querySelectorAll('[data-airi-dismiss]').forEach((control) => {
        control.addEventListener('click', () => {
            document.querySelector(control.dataset.airiDismiss)?.remove();
        });
    });
}

function initialiseRecognitionLog() {
    const logList = document.querySelector('#live-recognition-log');
    if (!logList) {
        return;
    }

    try {
        const socket = new WebSocket(`ws://${window.location.host}/ws/recognition/`);
        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const item = document.createElement('li');
                item.className = 'list-group-item d-flex justify-content-between align-items-center';
                item.innerHTML = `
                    <div>
                        <strong>${data.name}</strong>
                        <small class="d-block text-muted">${data.department}</small>
                    </div>
                    <span class="badge bg-${statusToBadge(data.status)}">${data.action}</span>
                `;
                logList.prepend(item);
                while (logList.children.length > 10) {
                    logList.removeChild(logList.lastChild);
                }
            } catch (error) {
                console.debug('WebSocket message parse error:', error);
            }
        };
        socket.onerror = () => console.debug('WebSocket not available (optional feature)');
    } catch (error) {
        console.debug('WebSocket not supported or disabled');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initialiseTheme();
    initialiseEvidenceModal();
    initialiseEvidenceControls();
    initialiseAttendanceFilters();
    initialiseMobileNavigation();
    initialiseDismissControls();
    initialiseRecognitionLog();
});
