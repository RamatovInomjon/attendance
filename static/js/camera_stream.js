/**
 * Two-camera live console.
 *
 * Camera transport remains `/ws/camera/{id}/`; frames remain JSON-wrapped
 * base64 JPEGs drawn to canvas. Recognition panels retain 10-second polling.
 */
(function initialiseLiveConsole() {
    'use strict';

    const MAX_RECENT_FEED_NODES = 30;
    const RECONNECT_DELAY_MS = 3000;

    function createElement(tagName, className, text) {
        const node = document.createElement(tagName);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = String(text);
        return node;
    }

    function safeMediaSource(value) {
        if (typeof value !== 'string' || !value.trim()) return '';
        const source = value.trim();
        if (source.startsWith('/') && !source.startsWith('//')) return source;
        try {
            const parsed = new URL(source, window.location.origin);
            if (parsed.origin === window.location.origin && ['http:', 'https:'].includes(parsed.protocol)) {
                return parsed.href;
            }
        } catch (error) {
            return '';
        }
        return '';
    }

    function formatTime(value) {
        if (!value) return '—';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function formatConfidence(value) {
        const score = Number(value);
        if (!Number.isFinite(score)) return 'Mavjud emas';
        return `${(score <= 1 ? score * 100 : score).toFixed(1)}%`;
    }

    function normalizeSnapshotPath(value) {
        if (typeof value !== 'string' || !value.trim()) return null;
        const source = value.trim();
        if (source.startsWith('/') || /^https?:\/\//i.test(source)) return source;
        return `/media/${source.replace(/^\/+/, '')}`;
    }

    function normalizeAttendanceEvent(data) {
        if (!data || data.type !== 'event') return null;
        const transition = typeof data.transition === 'string' ? data.transition : '';
        const action = transition === 'CHECK_IN'
            ? 'IN'
            : transition === 'CHECK_OUT'
                ? 'OUT'
                : data.role === 'IN' || data.role === 'OUT' ? data.role : '—';
        return {
            name: typeof data.name === 'string' ? data.name : "Noma'lum xodim",
            department: typeof data.department === 'string' ? data.department : '',
            camera: typeof data.camera === 'string' ? data.camera : '',
            action,
            transition,
            score: data.score,
            time: data.ts || data.time || '',
            snapshot: normalizeSnapshotPath(data.snapshot),
        };
    }

    function appendEvidence(container, source, title, subtitle, altText, fallbackIcon) {
        const safeSource = safeMediaSource(source);
        if (!safeSource) {
            const placeholder = createElement('div', 'live-evidence-placeholder');
            placeholder.setAttribute('aria-label', 'Dalil mavjud emas');
            const icon = createElement('i', `bi ${fallbackIcon}`);
            icon.setAttribute('aria-hidden', 'true');
            placeholder.append(icon);
            container.append(placeholder);
            return;
        }

        const button = createElement('button', 'live-evidence-button');
        button.type = 'button';
        button.dataset.airiEvidence = '';
        button.dataset.evidenceSrc = safeSource;
        button.dataset.evidenceTitle = title || 'Dalil';
        button.dataset.evidenceSubtitle = subtitle || '';
        const image = createElement('img');
        image.src = safeSource;
        image.alt = altText;
        image.loading = 'lazy';
        button.append(image);
        container.append(button);
    }

    function renderRecentEvents(container, entries) {
        if (!container) return;
        const fragment = document.createDocumentFragment();
        const recent = Array.isArray(entries) ? entries.slice(0, MAX_RECENT_FEED_NODES) : [];

        recent.forEach((entry) => {
            const card = createElement('article', 'live-event-card');
            const action = entry.action || entry.transition || '—';
            const subtitle = [entry.camera, action, entry.time].filter(Boolean).join(' · ');
            appendEvidence(
                card, entry.snapshot, entry.name || 'Tanish', subtitle,
                `${entry.name || 'Tanish'} uchun quality-best dalil`, 'bi-person-bounding-box',
            );

            const body = createElement('div', 'live-event-card__body');
            const heading = createElement('div', 'd-flex justify-content-between gap-2');
            const identity = createElement('div');
            identity.append(
                createElement('strong', '', entry.name || "Noma'lum xodim"),
                createElement('small', '', entry.department || "Bo'lim berilmagan"),
            );
            const badgeClass = action === 'IN' ? 'status-chip status-chip--success' : 'status-chip status-chip--warning';
            heading.append(identity, createElement('span', badgeClass, action));

            const details = createElement('dl');
            [
                ['Ishonch', formatConfidence(entry.score)],
                ['Kamera', entry.camera || '—'],
                ['Vaqt', entry.time || '—'],
            ].forEach(([label, value]) => {
                const item = createElement('div');
                item.append(createElement('dt', '', label), createElement('dd', '', value));
                details.append(item);
            });
            body.append(heading, details);
            card.append(body);
            fragment.append(card);
        });

        if (!recent.length) {
            const empty = createElement('div', 'live-empty-state');
            empty.dataset.feedEmpty = '';
            const icon = createElement('i', 'bi bi-person-check');
            icon.setAttribute('aria-hidden', 'true');
            empty.append(icon, createElement('p', 'mb-0', "Hozircha tanishlar yo'q."));
            fragment.append(empty);
        }
        container.replaceChildren(fragment);
        const count = document.getElementById('recentEventCount');
        if (count) count.textContent = String(recent.length);
    }

    function renderUnknownActivity(container, entries) {
        if (!container) return;
        const fragment = document.createDocumentFragment();
        const recent = Array.isArray(entries) ? entries.slice(0, MAX_RECENT_FEED_NODES) : [];

        recent.forEach((entry) => {
            const card = createElement('article', 'live-unknown-card');
            const title = `Noma'lum #${entry.id ?? '?'}`;
            const camera = entry.camera || (entry.camera_id ? `Kamera #${entry.camera_id}` : '—');
            const lastSeen = entry.last_seen_label || formatTime(entry.last_seen || entry.timestamp);
            appendEvidence(
                card, entry.snapshot, title, `${camera} · ${lastSeen}`,
                "Noma'lum shaxs uchun quality-best dalil", 'bi-person-exclamation',
            );
            const body = createElement('div');
            body.append(
                createElement('strong', '', title),
                createElement('small', '', `${camera} · ${lastSeen}`),
                createElement('span', '', `${entry.attempt_count ?? 0} kadr`),
            );
            card.append(body);
            fragment.append(card);
        });

        if (!recent.length) {
            const empty = createElement('div', 'live-empty-state');
            empty.dataset.feedEmpty = '';
            const icon = createElement('i', 'bi bi-shield-check');
            icon.setAttribute('aria-hidden', 'true');
            empty.append(icon, createElement('p', 'mb-0', "Noma'lum faollik qayd etilmadi."));
            fragment.append(empty);
        }
        container.replaceChildren(fragment);
    }

    class CameraStreamManager {
        constructor() {
            this.streams = new Map();
        }

        connect(card) {
            const cameraId = card.dataset.cameraId;
            const canvas = card.querySelector('[data-stream-canvas]');
            const streamEndpoint = canvas?.dataset.streamEndpoint;
            if (!cameraId || !canvas || !streamEndpoint || this.streams.has(cameraId)) return;

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/camera/${streamEndpoint}/`;
            const stream = {
                card, canvas, streamEndpoint, socket: null, reconnectTimer: null,
                closed: false, frameSequence: 0,
            };
            this.streams.set(cameraId, stream);

            const connectSocket = () => {
                if (stream.closed) return;
                const socket = new WebSocket(wsUrl);
                stream.socket = socket;

                socket.addEventListener('open', () => {
                    this.updateState(stream, 'unavailable', 'Kadr kutilmoqda');
                });
                socket.addEventListener('message', (event) => this.decodeFrame(stream, event.data));
                socket.addEventListener('error', () => this.updateState(stream, 'unavailable', 'Mavjud emas'));
                socket.addEventListener('close', () => {
                    if (stream.closed) return;
                    this.updateState(stream, 'unavailable', 'Mavjud emas');
                    if (stream.reconnectTimer === null) {
                        stream.reconnectTimer = window.setTimeout(() => {
                            stream.reconnectTimer = null;
                            connectSocket();
                        }, RECONNECT_DELAY_MS);
                    }
                });
            };

            stream.connectSocket = connectSocket;
            connectSocket();
        }

        decodeFrame(stream, rawData) {
            try {
                const data = JSON.parse(rawData);
                if (data.type !== 'frame' || !data.data) return;
                const frameSequence = ++stream.frameSequence;
                if (data.stale === true) {
                    this.updateState(stream, 'unavailable', 'Oqim eskirgan');
                    return;
                }
                const image = new Image();
                image.addEventListener('load', () => {
                    if (frameSequence !== stream.frameSequence) return;
                    const context = stream.canvas.getContext('2d');
                    if (stream.canvas.width !== image.width || stream.canvas.height !== image.height) {
                        stream.canvas.width = image.width;
                        stream.canvas.height = image.height;
                    }
                    context.drawImage(image, 0, 0);
                    stream.card.querySelector('[data-stream-placeholder]')?.setAttribute('hidden', '');
                    const fps = stream.card.querySelector('[data-camera-metric="camera-fps"]');
                    const lastFrame = stream.card.querySelector('[data-camera-metric="last-frame"]');
                    const latency = stream.card.querySelector('[data-camera-metric="latency"]');
                    if (fps && data.fps !== undefined) {
                        const frameFps = Number(data.fps);
                        fps.textContent = Number.isFinite(frameFps) && frameFps > 0
                            ? String(data.fps) : 'Mavjud emas';
                    }
                    if (lastFrame) lastFrame.textContent = new Date().toLocaleTimeString();
                    if (latency && data.delay_ms !== undefined) latency.textContent = `${data.delay_ms} ms`;
                    this.updateState(stream, 'online', 'Onlayn');
                }, { once: true });
                image.src = `data:image/jpeg;base64,${data.data}`;
            } catch (error) {
                console.warn(`Kamera ${stream.streamEndpoint} kadri o'qilmadi`, error);
            }
        }

        updateState(stream, state, label) {
            stream.card.dataset.streamState = state;
            const status = stream.card.querySelector('.live-camera-card__header [data-stream-state]');
            if (!status) return;
            status.dataset.streamState = state;
            status.textContent = label;
            status.classList.toggle('status-chip--success', state === 'online');
            status.classList.toggle('status-chip--warning', state === 'unavailable');
            status.classList.toggle('status-chip--danger', state === 'offline');
        }

        disconnect(cameraId) {
            const stream = this.streams.get(cameraId);
            if (!stream) return;
            stream.closed = true;
            if (stream.reconnectTimer !== null) window.clearTimeout(stream.reconnectTimer);
            stream.socket?.close();
            this.streams.delete(cameraId);
        }

        reconnectAll() {
            const cards = Array.from(document.querySelectorAll('[data-camera-id]'));
            Array.from(this.streams.keys()).forEach((cameraId) => this.disconnect(cameraId));
            cards.forEach((card) => this.connect(card));
        }

        disconnectAll() {
            Array.from(this.streams.keys()).forEach((cameraId) => this.disconnect(cameraId));
        }
    }

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { CameraStreamManager, normalizeAttendanceEvent };
    }
    if (typeof document === 'undefined') return;

    document.addEventListener('DOMContentLoaded', () => {
        const consoleElement = document.querySelector('[data-live-console]');
        if (!consoleElement) return;

        const eventFeed = consoleElement.querySelector('[data-recent-feed]');
        const unknownFeed = consoleElement.querySelector('[data-unknown-feed]');
        const eventsScript = document.getElementById('initial-recognition-events');
        const unknownScript = document.getElementById('initial-unknown-activity');
        let recentEvents = JSON.parse(eventsScript?.textContent || '[]');
        let unknownActivity = JSON.parse(unknownScript?.textContent || '[]');
        const streamManager = new CameraStreamManager();
        document.querySelectorAll('[data-camera-id]').forEach((card) => streamManager.connect(card));
        window.cameraStreamManager = streamManager;

        renderRecentEvents(eventFeed, recentEvents);
        renderUnknownActivity(unknownFeed, unknownActivity);

        const hydrateFromApi = (payload) => {
            if (Array.isArray(payload.events)) {
                recentEvents = payload.events.slice(0, MAX_RECENT_FEED_NODES);
                renderRecentEvents(eventFeed, recentEvents);
            }
            if (Array.isArray(payload.unknown_attempts)) {
                unknownActivity = payload.unknown_attempts.slice(0, MAX_RECENT_FEED_NODES);
                renderUnknownActivity(unknownFeed, unknownActivity);
            }
            if (!payload.stats) return;
            const mappings = [
                ['knownCount', payload.stats.recognized_today],
                ['checkedInCount', payload.stats.checked_in_today],
                ['checkedOutCount', payload.stats.checked_out_today],
                ['unknownCount', payload.stats.unknown_attempts],
                ['attendanceSummaryText', payload.stats.summary],
            ];
            mappings.forEach(([id, value]) => {
                const target = document.getElementById(id);
                if (target && value !== undefined && value !== null) target.textContent = String(value);
            });
        };

        const refreshLogs = async () => {
            try {
                const response = await fetch(consoleElement.dataset.recognitionLogsUrl, {
                    headers: { Accept: 'application/json' },
                });
                if (response.ok) hydrateFromApi(await response.json());
            } catch (error) {
                console.warn('Jonli qaydlarni yangilab bo\'lmadi', error);
            }
        };

        document.addEventListener('click', (event) => {
            const button = event.target.closest('[data-action]');
            if (!button || !consoleElement.contains(button)) return;
            if (button.dataset.action === 'reconnect-streams') {
                streamManager.reconnectAll();
            } else if (button.dataset.action === 'refresh-live-feed') {
                refreshLogs();
            } else if (button.dataset.action === 'fullscreen') {
                const canvas = button.closest('[data-camera-id]')?.querySelector('[data-stream-canvas]');
                if (!canvas) return;
                if (document.fullscreenElement) document.exitFullscreen();
                else canvas.requestFullscreen().catch((error) => console.warn("To'liq ekran ochilmadi", error));
            }
        });

        document.addEventListener('fullscreenchange', () => {
            document.querySelectorAll('[data-action="fullscreen"] span').forEach((label) => {
                label.textContent = document.fullscreenElement ? 'Yopish' : "To'liq ekran";
            });
        });

        let recognitionSocket = null;
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            recognitionSocket = new WebSocket(`${protocol}//${window.location.host}/ws/attendance/`);
            recognitionSocket.addEventListener('message', (event) => {
                try {
                    const data = JSON.parse(event.data);
                    const nextEvent = normalizeAttendanceEvent(data);
                    if (nextEvent) {
                        recentEvents = [nextEvent, ...recentEvents].slice(0, MAX_RECENT_FEED_NODES);
                        renderRecentEvents(eventFeed, recentEvents);
                    }
                } catch (error) {
                    console.warn('WebSocket qaydi yaroqsiz', error);
                }
            });
        } catch (error) {
            console.warn('WebSocket mavjud emas', error);
        }

        refreshLogs();
        const pollingTimer = window.setInterval(refreshLogs, 10000);
        window.addEventListener('beforeunload', () => {
            window.clearInterval(pollingTimer);
            recognitionSocket?.close();
            streamManager.disconnectAll();
        }, { once: true });
    });
})();
