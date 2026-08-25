"""
Network Diagnostics for RTSP Camera Connections
Migrated from recognition/network_diagnostics.py
"""
import logging
import socket
import subprocess
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def check_ip_reachability(ip: str, timeout: int = 3) -> Tuple[bool, str]:
    """Check if IP address is reachable using socket or ping."""
    # Try socket connection first
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, 80))
        sock.close()
        if result == 0 or result == 111:
            return True, f"✅ IP {ip} is reachable"
    except Exception:
        pass
    
    # Fallback to ping
    try:
        result = subprocess.run(
            ['ping', '-c', '1', '-w', str(timeout), ip],
            capture_output=True, text=True, timeout=timeout + 2
        )
        if result.returncode == 0:
            return True, f"✅ IP {ip} is reachable"
    except Exception:
        pass
    
    return False, f"❌ IP {ip} is not reachable"


def check_port_accessibility(ip: str, port: int, timeout: int = 3) -> Tuple[bool, str]:
    """Check if TCP port is accessible."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        
        if result == 0:
            return True, f"✅ Port {port} on {ip} is accessible"
        elif result == 111:
            return False, f"❌ Port {port} on {ip} is closed"
        else:
            return False, f"❌ Port {port} on {ip} is not accessible (error: {result})"
    except socket.timeout:
        return False, f"❌ Connection to {ip}:{port} timed out"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def parse_rtsp_url(rtsp_url: str) -> Optional[Tuple[str, int, str]]:
    """Parse RTSP URL to extract IP, port, and path."""
    try:
        parsed = urlparse(rtsp_url)
        ip = parsed.hostname
        port = parsed.port or 554
        path = parsed.path
        return (ip, port, path) if ip else None
    except Exception:
        return None


def diagnose_rtsp_connection(rtsp_url: str) -> dict:
    """Comprehensive RTSP connection diagnostics."""
    diagnostics = {
        "rtsp_url": rtsp_url,
        "valid_url": False,
        "ip_reachable": False,
        "port_accessible": False,
        "issues": [],
        "recommendations": []
    }
    
    parsed = parse_rtsp_url(rtsp_url)
    if not parsed:
        diagnostics["issues"].append("Invalid RTSP URL format")
        diagnostics["recommendations"].append(
            "Check RTSP URL format. Examples:\n"
            "  - Hikvision: rtsp://user:pass@192.168.1.100:554/Streaming/Channels/101\n"
            "  - Dahua: rtsp://user:pass@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
        )
        return diagnostics
    
    ip, port, path = parsed
    diagnostics.update({"valid_url": True, "ip": ip, "port": port, "path": path})
    
    # Check IP
    ip_reachable, ip_message = check_ip_reachability(ip)
    diagnostics["ip_reachable"] = ip_reachable
    diagnostics["ip_message"] = ip_message
    
    if not ip_reachable:
        diagnostics["issues"].append(f"IP {ip} is not reachable")
        diagnostics["recommendations"].extend([
            "1. Check if camera is powered on",
            "2. Verify camera IP address",
            "3. Check network cable connection"
        ])
        return diagnostics
    
    # Check port
    port_accessible, port_message = check_port_accessibility(ip, port)
    diagnostics["port_accessible"] = port_accessible
    diagnostics["port_message"] = port_message
    
    if not port_accessible:
        diagnostics["issues"].append(f"RTSP port {port} is not accessible")
        diagnostics["recommendations"].extend([
            "1. Verify RTSP service is enabled",
            "2. Check firewall rules (port 554)",
            "3. Verify camera credentials"
        ])
        return diagnostics
    
    diagnostics["recommendations"].append("✅ Network connectivity OK")
    return diagnostics
