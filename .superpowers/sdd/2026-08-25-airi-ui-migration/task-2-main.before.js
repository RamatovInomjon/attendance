document.addEventListener('DOMContentLoaded', () => {
    const logList = document.querySelector('#live-recognition-log');
    if (logList) {
        // WebSocket is optional - fail silently if not available
        try {
            const socket = new WebSocket(`ws://${window.location.host}/ws/recognition/`);

            socket.onopen = () => {
                console.log('WebSocket connected');
            };

            socket.onerror = (error) => {
                // Silently fail - WebSocket is optional for real-time updates
                // The recognition feed will still work via polling
                console.debug('WebSocket not available (optional feature)');
            };

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
                } catch (e) {
                    console.debug('WebSocket message parse error:', e);
                }
            };

            socket.onclose = () => {
                console.debug('WebSocket closed');
            };
        } catch (error) {
            // WebSocket is not available - this is OK, we use polling instead
            console.debug('WebSocket not supported or disabled');
        }
    }
});

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

