import { useState, useEffect, useRef } from 'react';
import { X, Download, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../api';

interface LogViewerProps {
    isOpen: boolean;
    onClose: () => void;
    darkMode: boolean;
}

export function LogViewer({ isOpen, onClose, darkMode }: LogViewerProps) {
    const [logs, setLogs] = useState<string[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [autoScroll, setAutoScroll] = useState(true);
    const logEndRef = useRef<HTMLDivElement>(null);
    const eventSourceRef = useRef<EventSource | null>(null);

    useEffect(() => {
        if (isOpen) {
            loadLogs();
            startLogStream();
        } else {
            stopLogStream();
        }

        return () => {
            stopLogStream();
        };
    }, [isOpen]);

    useEffect(() => {
        if (autoScroll && logEndRef.current) {
            logEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs, autoScroll]);

    const loadLogs = async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${api.getBaseUrl()}/logs?lines=500`);
            const data = await response.json();
            if (data.logs) {
                setLogs(data.logs);
            }
        } catch (error) {
            console.error('Failed to load logs:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const startLogStream = () => {
        // Subscribe to log events via SSE
        const eventSource = new EventSource(`${api.getBaseUrl()}/events`);
        eventSourceRef.current = eventSource;

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.level && data.message) {
                    // Format log entry
                    const logEntry = `[${data.level}] ${data.message}`;
                    setLogs(prev => [...prev.slice(-999), logEntry]); // Keep last 1000 lines
                }
            } catch (e) {
                // Handle non-JSON messages
            }
        };

        eventSource.addEventListener('log', (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.message) {
                    const logEntry = `[${data.level || 'INFO'}] ${data.message}`;
                    setLogs(prev => [...prev.slice(-999), logEntry]);
                }
            } catch (e) {
                // Ignore parse errors
            }
        });

        eventSource.onerror = () => {
            // Reconnect on error
            setTimeout(() => {
                if (isOpen && eventSourceRef.current) {
                    stopLogStream();
                    startLogStream();
                }
            }, 1000);
        };
    };

    const stopLogStream = () => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
    };

    const downloadLogs = () => {
        const logText = logs.join('\n');
        const blob = new Blob([logText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `heartmula-logs-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const getLogLevelColor = (line: string) => {
        if (line.includes('[ERROR]') || line.includes('[CRITICAL]')) {
            return darkMode ? 'text-red-400' : 'text-red-600';
        }
        if (line.includes('[WARNING]') || line.includes('[WARN]')) {
            return darkMode ? 'text-yellow-400' : 'text-yellow-600';
        }
        if (line.includes('[INFO]')) {
            return darkMode ? 'text-blue-400' : 'text-blue-600';
        }
        if (line.includes('[DEBUG]')) {
            return darkMode ? 'text-gray-400' : 'text-gray-600';
        }
        return darkMode ? 'text-gray-300' : 'text-gray-700';
    };

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
                {/* Backdrop */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-black/70 backdrop-blur-sm"
                    onClick={onClose}
                />

                {/* Modal */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className={`relative w-full max-w-4xl h-[80vh] rounded-xl shadow-2xl overflow-hidden flex flex-col ${
                        darkMode ? 'bg-[#181818]' : 'bg-white'
                    }`}
                >
                    {/* Header */}
                    <div className={`flex items-center justify-between px-6 py-4 border-b ${
                        darkMode ? 'border-[#282828]' : 'border-slate-200'
                    }`}>
                        <div className="flex items-center gap-3">
                            <h2 className={`text-xl font-semibold ${
                                darkMode ? 'text-white' : 'text-gray-900'
                            }`}>
                                Console Logs
                            </h2>
                            <div className="flex items-center gap-2">
                                <label className={`flex items-center gap-2 text-sm ${
                                    darkMode ? 'text-gray-400' : 'text-gray-600'
                                }`}>
                                    <input
                                        type="checkbox"
                                        checked={autoScroll}
                                        onChange={(e) => setAutoScroll(e.target.checked)}
                                        className="rounded"
                                    />
                                    Auto-scroll
                                </label>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={loadLogs}
                                disabled={isLoading}
                                className={`p-2 rounded-lg transition-colors ${
                                    darkMode
                                        ? 'hover:bg-[#282828] text-gray-400 hover:text-white'
                                        : 'hover:bg-gray-100 text-gray-600 hover:text-gray-900'
                                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                title="Refresh logs"
                            >
                                <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                            </button>
                            <button
                                onClick={downloadLogs}
                                className={`p-2 rounded-lg transition-colors ${
                                    darkMode
                                        ? 'hover:bg-[#282828] text-gray-400 hover:text-white'
                                        : 'hover:bg-gray-100 text-gray-600 hover:text-gray-900'
                                }`}
                                title="Download logs"
                            >
                                <Download className="w-5 h-5" />
                            </button>
                            <button
                                onClick={onClose}
                                className={`p-2 rounded-lg transition-colors ${
                                    darkMode
                                        ? 'hover:bg-[#282828] text-gray-400 hover:text-white'
                                        : 'hover:bg-gray-100 text-gray-600 hover:text-gray-900'
                                }`}
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                    </div>

                    {/* Log Content */}
                    <div className={`flex-1 overflow-y-auto p-4 font-mono text-sm ${
                        darkMode ? 'bg-[#0f0f0f]' : 'bg-gray-50'
                    }`}>
                        {isLoading && logs.length === 0 ? (
                            <div className={`text-center py-8 ${
                                darkMode ? 'text-gray-400' : 'text-gray-600'
                            }`}>
                                Loading logs...
                            </div>
                        ) : logs.length === 0 ? (
                            <div className={`text-center py-8 ${
                                darkMode ? 'text-gray-400' : 'text-gray-600'
                            }`}>
                                No logs available
                            </div>
                        ) : (
                            logs.map((line, index) => (
                                <div
                                    key={index}
                                    className={`py-1 ${getLogLevelColor(line)}`}
                                >
                                    {line}
                                </div>
                            ))
                        )}
                        <div ref={logEndRef} />
                    </div>

                    {/* Footer */}
                    <div className={`px-6 py-3 border-t ${
                        darkMode ? 'border-[#282828] bg-[#181818]' : 'border-slate-200 bg-white'
                    }`}>
                        <div className={`text-xs ${
                            darkMode ? 'text-gray-500' : 'text-gray-500'
                        }`}>
                            {logs.length} log entries • Real-time updates enabled
                        </div>
                    </div>
                </motion.div>
            </div>
        </AnimatePresence>
    );
}
