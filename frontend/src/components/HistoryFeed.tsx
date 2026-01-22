import React, { useEffect, useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api, type Job } from '../api';
import { AlertCircle, Edit2, Check, Trash2, ArrowRightCircle, RefreshCcw, RotateCw, Play, Pause, Volume2, Music, Heart, ListPlus, ChevronLeft, ChevronRight } from 'lucide-react';
import { AlbumCover } from './AlbumCover';

// Animated equalizer bars component
const EqualizerBars: React.FC<{ darkMode?: boolean }> = ({ darkMode }) => (
    <div className="flex items-end gap-[3px] h-5">
        {[0, 1, 2, 3, 4].map((i) => (
            <motion.div
                key={i}
                className={`w-[3px] rounded-full ${darkMode ? 'bg-[#1DB954]' : 'bg-gradient-to-t from-cyan-500 to-purple-500'}`}
                animate={{
                    height: ['30%', '100%', '50%', '80%', '30%'],
                }}
                transition={{
                    duration: 0.6 + i * 0.1,
                    repeat: Infinity,
                    delay: i * 0.08,
                    ease: 'easeInOut',
                }}
            />
        ))}
    </div>
);

// Shimmer effect overlay
const ShimmerOverlay: React.FC<{ darkMode?: boolean }> = ({ darkMode }) => (
    <motion.div
        className="absolute inset-0 rounded-md overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
    >
        <motion.div
            className={`absolute inset-0 ${
                darkMode
                    ? 'bg-gradient-to-r from-transparent via-white/10 to-transparent'
                    : 'bg-gradient-to-r from-transparent via-white/40 to-transparent'
            }`}
            animate={{ x: ['-100%', '100%'] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'linear', repeatDelay: 0.5 }}
        />
    </motion.div>
);

interface HistoryFeedProps {
    history: Job[];
    currentJobId: string | null;
    onRefresh: () => void;
    onExtend?: (job: Job) => void;
    onReimport?: (job: Job) => void;
    onRetry?: (job: Job) => void;
    onPlayTrack?: (job: Job) => void;
    onPauseTrack?: () => void;
    playingTrackId?: string;
    isTrackPlaying?: boolean;
    queuedJobs?: Map<string, number>; // job_id -> queue position
    darkMode?: boolean;
    likedIds?: Set<string>;
    onToggleLike?: (jobId: string, isLiked: boolean) => void;
    onAddToPlaylist?: (job: Job) => void;
    onSelectTrack?: (job: Job) => void;
    selectedTrackId?: string;
}

const ITEMS_PER_PAGE = 100;

export const HistoryFeed: React.FC<HistoryFeedProps> = ({ history, currentJobId, onRefresh, onExtend, onReimport, onRetry, onPlayTrack, onPauseTrack, playingTrackId, isTrackPlaying = false, queuedJobs, darkMode = false, likedIds = new Set(), onToggleLike, onAddToPlaylist, onSelectTrack, selectedTrackId }) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [tempTitle, setTempTitle] = useState("");
    const [jobProgress, setJobProgress] = useState<Map<string, { progress: number; msg: string }>>(new Map());
    const [stuckJobs, setStuckJobs] = useState<Set<string>>(new Set());
    const progressRef = useRef<Map<string, { progress: number; timestamp: number }>>(new Map());
    const [currentPage, setCurrentPage] = useState(0);

    // Pagination
    const totalPages = Math.ceil(history.length / ITEMS_PER_PAGE);
    const paginatedHistory = useMemo(() => {
        const start = currentPage * ITEMS_PER_PAGE;
        return history.slice(start, start + ITEMS_PER_PAGE);
    }, [history, currentPage]);

    // Reset to first page when history changes significantly
    useEffect(() => {
        if (currentPage >= totalPages && totalPages > 0) {
            setCurrentPage(totalPages - 1);
        }
    }, [totalPages, currentPage]);

    useEffect(() => {
        const handleProgress = (e: any) => {
            const data = e.detail;
            if (data.progress !== undefined && data.job_id) {
                // Track progress per job
                setJobProgress(prev => {
                    const next = new Map(prev);
                    next.set(data.job_id, { progress: data.progress, msg: data.msg || '' });
                    return next;
                });
                // Track progress for stuck detection
                progressRef.current.set(data.job_id, { progress: data.progress, timestamp: Date.now() });
                // If progress > 0, remove from stuck
                if (data.progress > 0) {
                    setStuckJobs(prev => {
                        const next = new Set(prev);
                        next.delete(data.job_id);
                        return next;
                    });
                }
            }
        };
        window.addEventListener('heartmula_progress', handleProgress);
        return () => window.removeEventListener('heartmula_progress', handleProgress);
    }, []);

    // Clean up progress data for completed/failed jobs
    useEffect(() => {
        history.forEach(job => {
            if (job.status === 'completed' || job.status === 'failed') {
                // Remove progress data for finished jobs
                if (jobProgress.has(job.id)) {
                    setJobProgress(prev => {
                        const next = new Map(prev);
                        next.delete(job.id);
                        return next;
                    });
                }
                progressRef.current.delete(job.id);
                setStuckJobs(prev => {
                    if (prev.has(job.id)) {
                        const next = new Set(prev);
                        next.delete(job.id);
                        return next;
                    }
                    return prev;
                });
            }
        });
    }, [history]);

    // Check for stuck jobs every 5 seconds
    useEffect(() => {
        const checkStuck = () => {
            const now = Date.now();
            const STUCK_THRESHOLD_MS = 15000; // 15 seconds at 0%

            history.forEach(job => {
                if (job.status === 'processing' || job.status === 'queued') {
                    const tracked = progressRef.current.get(job.id);
                    // If we have no progress data or progress is 0 for too long
                    if (!tracked || (tracked.progress === 0 && now - tracked.timestamp > STUCK_THRESHOLD_MS)) {
                        // Check if job was created more than threshold ago
                        const createdAt = new Date(job.created_at + "Z").getTime();
                        if (now - createdAt > STUCK_THRESHOLD_MS) {
                            setStuckJobs(prev => new Set(prev).add(job.id));
                        }
                    }
                }
            });
        };

        // Initial check
        checkStuck();
        const interval = setInterval(checkStuck, 5000);
        return () => clearInterval(interval);
    }, [history]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = 0;
        }
    }, [history.length, currentJobId]);

    const handleRenameStart = (job: Job) => {
        setEditingId(job.id);
        setTempTitle(job.title || job.prompt || "Untitled");
    };

    const handleRenameSave = async (jobId: string) => {
        if (!tempTitle.trim()) return;
        try {
            await api.renameJob(jobId, tempTitle);
            setEditingId(null);
        } catch (e) {
            console.error("Rename failed", e);
        }
    };

    const handleDelete = async (jobId: string) => {
        if (!confirm("Are you sure you want to delete this track? This action cannot be undone.")) return;
        try {
            await api.deleteJob(jobId);
            onRefresh();
        } catch (e) {
            console.error("Delete failed", e);
            alert("Failed to delete track");
        }
    };

    // Dark mode classes - Spotify style
    const textClass = darkMode ? 'text-white' : 'text-slate-900';
    const mutedTextClass = darkMode ? 'text-[#b3b3b3]' : 'text-slate-500';

    // Only show as generating if actually processing/queued AND not failed
    const isGenerating = (job: Job) => {
        if (job.status === 'failed' || job.status === 'completed' || job.error_msg) {
            return false;
        }
        return job.status === 'processing' || job.status === 'queued';
    };

    // Format duration from ms to mm:ss
    const formatDuration = (ms?: number) => {
        if (!ms) return null;
        const totalSeconds = Math.floor(ms / 1000);
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    };

    // Card content component to avoid duplication
    const CardContent: React.FC<{ job: Job }> = ({ job }) => (
        <div className="p-3 sm:p-5">
            {/* Header Row: Status, Title, Meta */}
            <div className="flex justify-between items-start gap-2 sm:gap-4">
                <div className="flex items-center gap-2 sm:gap-4 min-w-0 flex-1">
                    {/* Album Cover with animations */}
                    <div className="relative group/cover shrink-0">
                        {isGenerating(job) ? (
                            <div className="relative">
                                <motion.div
                                    animate={{ scale: [1, 1.02, 1] }}
                                    transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                                >
                                    <AlbumCover seed={job.id} size="md" />
                                </motion.div>
                                <ShimmerOverlay darkMode={darkMode} />
                                {/* Pulsing ring */}
                                <motion.div
                                    className={`absolute -inset-1 rounded-lg ${darkMode ? 'border-[#1DB954]' : 'border-cyan-500'} border-2`}
                                    animate={{ opacity: [0.3, 0.8, 0.3], scale: [1, 1.05, 1] }}
                                    transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                                />
                            </div>
                        ) : (
                            <AlbumCover seed={job.id} size="md" className={job.status === 'failed' ? 'opacity-50 grayscale' : ''} />
                        )}

                        {/* Play/Pause Button Overlay */}
                        {job.status === 'completed' && job.audio_path && (
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (playingTrackId === job.id && isTrackPlaying) {
                                        onPauseTrack?.();
                                    } else {
                                        onPlayTrack?.(job);
                                    }
                                }}
                                className={`absolute inset-0 flex items-center justify-center rounded-md transition-all duration-200 ${
                                    playingTrackId === job.id && isTrackPlaying
                                        ? 'bg-black/40'
                                        : 'bg-black/0 hover:bg-black/40 opacity-0 group-hover/cover:opacity-100'
                                }`}
                            >
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center shadow-lg transition-transform hover:scale-110 ${
                                    darkMode ? 'bg-[#1DB954] text-black' : 'bg-white text-slate-900'
                                }`}>
                                    {playingTrackId === job.id && isTrackPlaying ? (
                                        <Pause className="w-4 h-4" fill="currentColor" />
                                    ) : (
                                        <Play className="w-4 h-4 ml-0.5" fill="currentColor" />
                                    )}
                                </div>
                            </button>
                        )}

                        {/* Status Indicator for failed */}
                        {job.status === 'failed' && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <AlertCircle className="w-5 h-5 text-red-500" />
                            </div>
                        )}

                        {/* Now Playing Indicator */}
                        {playingTrackId === job.id && isTrackPlaying && (
                            <div className="absolute -bottom-1 -right-1">
                                <div className={`w-4 h-4 rounded-full flex items-center justify-center ${darkMode ? 'bg-[#1DB954]' : 'bg-green-500'}`}>
                                    <Volume2 className="w-2.5 h-2.5 text-white" />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Title & Tags */}
                    <div className="min-w-0 flex-1">
                        {editingId === job.id ? (
                            <div className="flex items-center gap-2">
                                <input
                                    autoFocus
                                    className={`text-base font-bold border rounded-sm px-2 py-0.5 focus:outline-none focus:ring-2 ring-cyan-500 w-full font-mono ${darkMode ? 'bg-slate-800 border-cyan-500/50 text-white' : 'bg-white border-cyan-300 text-slate-800'}`}
                                    value={tempTitle}
                                    onChange={e => setTempTitle(e.target.value)}
                                    onKeyDown={e => {
                                        if (e.key === 'Enter') handleRenameSave(job.id);
                                        if (e.key === 'Escape') setEditingId(null);
                                    }}
                                    onBlur={() => setEditingId(null)}
                                />
                                <button
                                    onMouseDown={e => e.preventDefault()}
                                    onClick={() => handleRenameSave(job.id)}
                                    className={`p-1 rounded-full transition-colors ${darkMode ? 'bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30' : 'bg-cyan-100 text-cyan-700 hover:bg-cyan-200'}`}
                                >
                                    <Check className="w-4 h-4" />
                                </button>
                            </div>
                        ) : (
                            <div className="flex items-center gap-2 group/title">
                                <h3
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        if (!isGenerating(job)) handleRenameStart(job);
                                    }}
                                    className={`text-base font-semibold truncate transition-colors ${
                                        isGenerating(job)
                                            ? darkMode ? 'text-[#1DB954]' : 'text-transparent bg-clip-text bg-gradient-to-r from-cyan-500 to-purple-500'
                                            : playingTrackId === job.id
                                                ? darkMode ? 'text-[#1DB954]' : 'text-green-600'
                                                : darkMode ? 'text-white hover:underline cursor-pointer' : 'text-slate-800 hover:text-cyan-600 cursor-pointer'
                                    }`}>
                                    {isGenerating(job) ? 'Creating your track...' : (job.title || job.prompt || "Untitled Track")}
                                </h3>
                                {!isGenerating(job) && (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleRenameStart(job);
                                        }}
                                        className={`p-1 opacity-0 group-hover/title:opacity-100 transition-opacity shrink-0 rounded hover:bg-black/10 ${darkMode ? 'text-[#b3b3b3] hover:text-white' : 'text-slate-300 hover:text-slate-600'}`}
                                    >
                                        <Edit2 className="w-3 h-3" />
                                    </button>
                                )}
                            </div>
                        )}

                        <div className="flex items-center gap-2 mt-0.5">
                            <p className={`text-sm truncate ${mutedTextClass}`}>
                                {job.tags || job.prompt}
                            </p>
                            {job.status === 'completed' && job.duration_ms && (
                                <>
                                    <span className={`text-sm ${darkMode ? 'text-[#535353]' : 'text-slate-300'}`}>•</span>
                                    <span className={`text-sm shrink-0 ${darkMode ? 'text-[#b3b3b3]' : 'text-slate-500'}`}>
                                        {formatDuration(job.duration_ms)}
                                    </span>
                                </>
                            )}
                        </div>

                        {/* Progress info for generating */}
                        {isGenerating(job) && (
                            <div className="mt-2 flex items-center gap-3">
                                <EqualizerBars darkMode={darkMode} />
                                <span className={`text-xs ${darkMode ? 'text-[#1DB954]' : 'text-cyan-600'}`}>
                                    {job.status === 'queued'
                                        ? `Queued${queuedJobs?.get(job.id) ? ` #${queuedJobs.get(job.id)}` : ''}`
                                        : (jobProgress.get(job.id)?.progress || 0) >= 100
                                            ? 'Finalizing...'
                                            : `${jobProgress.get(job.id)?.progress || 0}%`
                                    }
                                </span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Status & Actions */}
                <div className="text-right flex flex-col items-end gap-1 sm:gap-2 shrink-0">
                    <div className="flex items-center gap-1 sm:gap-2">
                        {/* Time - hidden on very small screens */}
                        <span className={`hidden xs:inline text-xs ${darkMode ? 'text-[#727272]' : 'text-slate-400'}`}>
                            {new Date(job.created_at + "Z").toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
                        </span>

                        {job.status === 'failed' && (
                            <span className={`text-xs ${darkMode ? 'text-red-400' : 'text-red-500'}`}>Failed</span>
                        )}

                        {/* Like Button - always visible if liked */}
                        {job.status === 'completed' && (
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onToggleLike?.(job.id, likedIds.has(job.id));
                                }}
                                className={`p-1.5 rounded-full transition-all ${
                                    likedIds.has(job.id)
                                        ? darkMode ? 'text-[#1DB954]' : 'text-red-500'
                                        : darkMode ? 'text-[#727272] hover:text-[#1DB954] sm:opacity-0 sm:group-hover:opacity-100' : 'text-slate-300 hover:text-red-500 sm:opacity-0 sm:group-hover:opacity-100'
                                }`}
                                title={likedIds.has(job.id) ? 'Remove from Liked Songs' : 'Add to Liked Songs'}
                            >
                                <Heart className="w-4 h-4" fill={likedIds.has(job.id) ? 'currentColor' : 'none'} />
                            </button>
                        )}

                        {/* Add to Playlist Button - hidden on mobile */}
                        {job.status === 'completed' && onAddToPlaylist && (
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onAddToPlaylist(job);
                                }}
                                className={`hidden sm:block p-1.5 rounded-full opacity-0 group-hover:opacity-100 transition-all ${darkMode ? 'text-[#727272] hover:text-white hover:bg-[#3E3E3E]' : 'text-slate-300 hover:text-slate-600'}`}
                                title="Add to Playlist"
                            >
                                <ListPlus className="w-4 h-4" />
                            </button>
                        )}

                        {/* Delete (not for generating) - hidden on mobile */}
                        {!isGenerating(job) && (
                            <button
                                onClick={() => handleDelete(job.id)}
                                className={`hidden sm:block p-1.5 rounded-full opacity-0 group-hover:opacity-100 transition-all ${darkMode ? 'text-[#727272] hover:text-white hover:bg-[#3E3E3E]' : 'text-slate-300 hover:text-red-500'}`}
                                title="Delete Track"
                            >
                                <Trash2 className="w-4 h-4" />
                            </button>
                        )}
                    </div>

                    {/* Actions: Extend & Reimport - show on hover, hidden on mobile */}
                    {job.status === 'completed' && (
                        <div className="hidden sm:flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            {onReimport && (
                                <button
                                    onClick={() => onReimport(job)}
                                    className={`text-xs flex items-center gap-1 px-3 py-1.5 rounded-full transition-colors ${darkMode ? 'text-white bg-[#3E3E3E] hover:bg-[#4D4D4D]' : 'text-slate-700 bg-slate-100 hover:bg-slate-200'}`}
                                    title="Load lyrics and style back into editor"
                                >
                                    <RefreshCcw className="w-3 h-3" />
                                    Reimport
                                </button>
                            )}
                            {onExtend && (
                                <button
                                    onClick={() => onExtend(job)}
                                    className={`text-xs flex items-center gap-1 px-3 py-1.5 rounded-full transition-colors ${darkMode ? 'text-black bg-white hover:bg-white/90' : 'text-white bg-slate-900 hover:bg-slate-800'}`}
                                    title="Extend this track with same seed"
                                >
                                    <ArrowRightCircle className="w-3 h-3" />
                                    Extend
                                </button>
                            )}
                        </div>
                    )}

                    {/* Retry for stuck jobs */}
                    {(isGenerating(job) && onRetry && stuckJobs.has(job.id)) && (
                        <button
                            onClick={() => onRetry(job)}
                            className={`text-xs flex items-center gap-1 px-3 py-1.5 rounded-full ${darkMode ? 'text-amber-400 bg-amber-900/30 hover:bg-amber-900/50' : 'text-amber-600 bg-amber-50 hover:bg-amber-100'}`}
                            title="Job appears stuck - click to retry"
                        >
                            <RotateCw className="w-3 h-3" />
                            Retry
                        </button>
                    )}
                </div>
            </div>

            {/* Progress Bar for generating */}
            {(job.status === 'processing' || job.status === 'queued') && (
                <div className="mt-4">
                    <div className={`h-1.5 rounded-full overflow-hidden ${darkMode ? 'bg-[#404040]' : 'bg-slate-200'}`}>
                        <motion.div
                            className={`h-full rounded-full ${
                                darkMode
                                    ? 'bg-[#1DB954]'
                                    : 'bg-gradient-to-r from-cyan-500 to-purple-500'
                            }`}
                            initial={{ width: '0%' }}
                            animate={{ width: `${Math.min(jobProgress.get(job.id)?.progress || 0, 100)}%` }}
                            transition={{ duration: 0.3, ease: 'easeOut' }}
                        />
                    </div>
                    <div className="flex items-center justify-between mt-1.5">
                        <p className={`text-xs ${darkMode ? 'text-[#727272]' : 'text-slate-400'}`}>
                            {(jobProgress.get(job.id)?.progress || 0) >= 100
                                ? 'Finalizing your track...'
                                : jobProgress.get(job.id)?.msg || (job.status === 'queued' ? 'Waiting in queue...' : 'Starting...')
                            }
                        </p>
                        <button
                            onClick={() => handleDelete(job.id)}
                            className={`text-xs px-2 py-1 rounded transition-colors ${darkMode ? 'text-red-400 hover:bg-red-500/20' : 'text-rose-500 hover:bg-rose-100'}`}
                            title="Delete stuck job"
                        >
                            <Trash2 className="w-3.5 h-3.5" />
                        </button>
                    </div>
                </div>
            )}

            {/* Error Message with Retry/Delete */}
            {job.status === 'failed' && (
                <div className={`mt-4 p-3 rounded-lg ${darkMode ? 'bg-[#282828]' : 'bg-rose-50/50 border border-rose-100'}`}>
                    <div className={`flex items-center gap-2 text-sm ${darkMode ? 'text-red-400' : 'text-rose-600'}`}>
                        <AlertCircle className="w-4 h-4 flex-shrink-0" />
                        <span className="flex-1">{job.error_msg || "Generation failed"}</span>
                    </div>
                    <div className="mt-3 flex gap-2">
                        {onRetry && (
                            <button
                                onClick={() => onRetry(job)}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${darkMode ? 'bg-white text-black hover:scale-105' : 'bg-slate-900 text-white hover:bg-slate-800'}`}
                            >
                                <RotateCw className="w-4 h-4" />
                                Try Again
                            </button>
                        )}
                        <button
                            onClick={() => handleDelete(job.id)}
                            className={`flex items-center justify-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${darkMode ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' : 'bg-rose-100 text-rose-600 hover:bg-rose-200'}`}
                        >
                            <Trash2 className="w-4 h-4" />
                            Delete
                        </button>
                    </div>
                </div>
            )}

            {/* Lyrics Accordion */}
            {job.lyrics && !isGenerating(job) && (
                <div className="mt-4">
                    <details className="group/lyrics">
                        <summary className={`text-xs cursor-pointer transition-colors list-none flex items-center gap-1 ${darkMode ? 'text-[#b3b3b3] hover:text-white' : 'text-slate-500 hover:text-slate-700'}`}>
                            <span>View lyrics</span>
                            <span className="group-open/lyrics:rotate-180 transition-transform text-[10px]">▼</span>
                        </summary>
                        <div className={`mt-2 text-sm whitespace-pre-line p-4 rounded-lg leading-relaxed ${darkMode ? 'text-[#b3b3b3] bg-[#282828]' : 'text-slate-600 bg-slate-50'}`}>
                            {job.lyrics}
                        </div>
                    </details>
                </div>
            )}
        </div>
    );

    return (
        <div className="h-full flex flex-col overflow-hidden relative">
            {/* Feed Header */}
            <div className={`p-4 sm:p-8 pb-4 sm:pb-6 flex items-end gap-4 ${darkMode ? 'bg-gradient-to-b from-[#1a1a1a] to-transparent' : ''}`}>
                <div>
                    <h1 className={`text-2xl sm:text-4xl font-bold ${textClass} tracking-tighter`}>Your Library</h1>
                    <p className={`${mutedTextClass} text-xs sm:text-sm mt-1`}>Recently created tracks</p>
                </div>
            </div>

            {/* List */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 sm:p-8 pt-0 space-y-2 sm:space-y-3 pb-48">
                <AnimatePresence initial={false}>
                    {paginatedHistory.map((job) => (
                        <motion.div
                            key={job.id}
                            initial={{ opacity: 0, y: 20, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
                            layout
                            transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
                        >
                            {isGenerating(job) ? (
                                /* Generating card with animated glow border */
                                <motion.div
                                    className="relative rounded-xl"
                                    animate={{
                                        boxShadow: darkMode
                                            ? ['0 0 20px rgba(29, 185, 84, 0.2)', '0 0 35px rgba(29, 185, 84, 0.4)', '0 0 20px rgba(29, 185, 84, 0.2)']
                                            : ['0 0 20px rgba(6, 182, 212, 0.2)', '0 0 35px rgba(139, 92, 246, 0.3)', '0 0 20px rgba(6, 182, 212, 0.2)']
                                    }}
                                    transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                                >
                                    {/* Animated gradient border */}
                                    <div className="absolute -inset-[1px] rounded-xl overflow-hidden">
                                        <motion.div
                                            className={`absolute inset-0 ${
                                                darkMode
                                                    ? 'bg-gradient-to-r from-[#1DB954] via-[#1ed760] to-[#1DB954]'
                                                    : 'bg-gradient-to-r from-cyan-500 via-purple-500 to-cyan-500'
                                            }`}
                                            animate={{ x: ['-100%', '100%'] }}
                                            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                                            style={{ width: '200%' }}
                                        />
                                    </div>
                                    {/* Inner card */}
                                    <div className={`relative rounded-xl overflow-hidden ${darkMode ? 'bg-[#181818]' : 'bg-white/95 backdrop-blur-md'}`}>
                                        <CardContent job={job} />
                                    </div>
                                </motion.div>
                            ) : (
                                /* Regular card */
                                <div
                                    className={`
                                        group relative overflow-hidden rounded-xl transition-all duration-200 cursor-pointer
                                        ${playingTrackId === job.id
                                            ? darkMode
                                                ? 'bg-[#282828] ring-2 ring-[#1DB954]'
                                                : 'bg-white/90 border border-green-400 shadow-xl shadow-green-500/10 ring-1 ring-green-400/30'
                                            : selectedTrackId === job.id
                                                ? darkMode
                                                    ? 'bg-[#282828] ring-1 ring-[#1DB954]/50'
                                                    : 'bg-white/80 border border-cyan-300 shadow-lg'
                                                : job.status === 'failed'
                                                    ? darkMode
                                                        ? 'bg-[#181818] border border-red-500/30'
                                                        : 'bg-white/40 border border-red-200 backdrop-blur-md'
                                                    : darkMode
                                                        ? 'bg-[#181818] hover:bg-[#282828]'
                                                        : 'bg-white/40 border border-slate-200/50 hover:bg-white/60 hover:border-slate-300 hover:shadow-lg backdrop-blur-md'
                                        }
                                    `}
                                    onClick={() => job.status === 'completed' && onSelectTrack?.(job)}
                                >
                                    <CardContent job={job} />
                                </div>
                            )}
                        </motion.div>
                    ))}
                </AnimatePresence>

                {history.length === 0 && (
                    <div className={`h-96 flex flex-col items-center justify-center ${darkMode ? 'text-[#727272]' : 'text-slate-300'}`}>
                        <div className={`w-20 h-20 rounded-full flex items-center justify-center mb-6 ${darkMode ? 'bg-[#282828]' : 'bg-slate-100'}`}>
                            <Music className={`w-10 h-10 ${darkMode ? 'text-[#1DB954]' : 'text-slate-400'}`} />
                        </div>
                        <p className={`text-lg font-semibold mb-2 ${darkMode ? 'text-white' : 'text-slate-700'}`}>No tracks yet</p>
                        <p className="text-sm">Create your first track to get started</p>
                    </div>
                )}

                {/* Pagination Controls */}
                {totalPages > 1 && (
                    <div className={`flex items-center justify-center gap-4 py-6 mt-4 mb-8 border-t ${darkMode ? 'border-[#282828]' : 'border-slate-200'}`}>
                        <button
                            onClick={() => {
                                setCurrentPage(p => Math.max(0, p - 1));
                                scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
                            }}
                            disabled={currentPage === 0}
                            className={`p-2 rounded-full transition-all disabled:opacity-30 disabled:cursor-not-allowed ${
                                darkMode
                                    ? 'bg-[#282828] text-white hover:bg-[#383838] disabled:hover:bg-[#282828]'
                                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:hover:bg-slate-100'
                            }`}
                        >
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                        <span className={`text-sm font-medium ${darkMode ? 'text-[#b3b3b3]' : 'text-slate-600'}`}>
                            Page {currentPage + 1} of {totalPages}
                            <span className={`ml-2 text-xs ${darkMode ? 'text-[#727272]' : 'text-slate-400'}`}>
                                ({history.length} tracks)
                            </span>
                        </span>
                        <button
                            onClick={() => {
                                setCurrentPage(p => Math.min(totalPages - 1, p + 1));
                                scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
                            }}
                            disabled={currentPage >= totalPages - 1}
                            className={`p-2 rounded-full transition-all disabled:opacity-30 disabled:cursor-not-allowed ${
                                darkMode
                                    ? 'bg-[#282828] text-white hover:bg-[#383838] disabled:hover:bg-[#282828]'
                                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:hover:bg-slate-100'
                            }`}
                        >
                            <ChevronRight className="w-5 h-5" />
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};
