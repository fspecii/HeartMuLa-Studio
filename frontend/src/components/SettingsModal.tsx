import { useState, useEffect } from 'react';
import { X, Cpu, RefreshCw, AlertTriangle, Check, Settings2, Globe, Key, CheckCircle, XCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { GPUStatus, StartupStatus, GPUSettings, LLMSettings } from '../api';

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    darkMode: boolean;
    gpuStatus: GPUStatus | null;
    currentSettings: GPUSettings | null;
    onSave: (settings: GPUSettings) => Promise<void>;
    onReload: (settings: GPUSettings) => Promise<void>;
    startupStatus: StartupStatus | null;
    llmSettings: LLMSettings | null;
    onSaveLLM: (settings: {
        ollama_host?: string;
        openrouter_api_key?: string;
        custom_api_base_url?: string;
        custom_api_key?: string;
        custom_api_model?: string;
    }) => Promise<void>;
}

export function SettingsModal({
    isOpen,
    onClose,
    darkMode,
    gpuStatus,
    currentSettings,
    onSave,
    onReload,
    startupStatus,
    llmSettings,
    onSaveLLM
}: SettingsModalProps) {
    const [settings, setSettings] = useState<GPUSettings>({
        quantization_4bit: 'auto',
        sequential_offload: 'auto',
        torch_compile: false,
        torch_compile_mode: 'default'
    });
    const [ollamaPreset, setOllamaPreset] = useState('host'); // 'host', 'localhost', 'custom'
    const [customOllamaUrl, setCustomOllamaUrl] = useState('');
    const [openrouterKey, setOpenrouterKey] = useState('');
    // Custom API state
    const [customApiBaseUrl, setCustomApiBaseUrl] = useState('');
    const [customApiKey, setCustomApiKey] = useState('');
    const [customApiModel, setCustomApiModel] = useState('');

    // Ollama presets
    const OLLAMA_PRESETS = {
        host: { label: 'On my computer', url: 'http://host.docker.internal:11434', desc: 'Ollama installed on your machine (most common)' },
        localhost: { label: 'Inside container', url: 'http://localhost:11434', desc: 'Ollama running inside this Docker container' },
        custom: { label: 'Custom URL', url: '', desc: 'External server or custom setup' }
    };
    const [isSaving, setIsSaving] = useState(false);
    const [isSavingLLM, setIsSavingLLM] = useState(false);
    const [isReloading, setIsReloading] = useState(false);
    const [hasChanges, setHasChanges] = useState(false);
    const [hasLLMChanges, setHasLLMChanges] = useState(false);
    const [saveSuccess, setSaveSuccess] = useState(false);
    const [saveLLMSuccess, setSaveLLMSuccess] = useState(false);

    // Load current settings when modal opens
    useEffect(() => {
        if (currentSettings) {
            setSettings(currentSettings);
        }
    }, [currentSettings, isOpen]);

    // Load LLM settings when modal opens
    useEffect(() => {
        if (llmSettings && isOpen) {
            const currentUrl = llmSettings.ollama_host || '';

            // Detect which preset matches
            if (currentUrl === OLLAMA_PRESETS.host.url || currentUrl === '') {
                setOllamaPreset('host');
            } else if (currentUrl === OLLAMA_PRESETS.localhost.url) {
                setOllamaPreset('localhost');
            } else {
                setOllamaPreset('custom');
                setCustomOllamaUrl(currentUrl);
            }

            // Don't load masked key - leave empty for new input
            if (!openrouterKey) {
                setOpenrouterKey('');
            }

            // Load Custom API settings
            setCustomApiBaseUrl(llmSettings.custom_api_base_url || '');
            setCustomApiModel(llmSettings.custom_api_model || '');
            // Don't load masked key - leave empty for new input
            if (!customApiKey) {
                setCustomApiKey('');
            }
        }
    }, [llmSettings, isOpen]);

    // Track GPU changes
    useEffect(() => {
        if (currentSettings) {
            const changed = JSON.stringify(settings) !== JSON.stringify(currentSettings);
            setHasChanges(changed);
        }
    }, [settings, currentSettings]);

    // Track LLM changes
    useEffect(() => {
        if (llmSettings) {
            // Calculate the effective URL based on preset
            const effectiveUrl = ollamaPreset === 'custom' ? customOllamaUrl : OLLAMA_PRESETS[ollamaPreset as keyof typeof OLLAMA_PRESETS].url;
            const hostChanged = effectiveUrl !== (llmSettings.ollama_host || OLLAMA_PRESETS.host.url);
            const keyChanged = openrouterKey !== '' && openrouterKey !== llmSettings.openrouter_api_key;
            // Track Custom API changes
            const customBaseUrlChanged = customApiBaseUrl !== (llmSettings.custom_api_base_url || '');
            const customModelChanged = customApiModel !== (llmSettings.custom_api_model || '');
            const customKeyChanged = customApiKey !== '' && customApiKey !== llmSettings.custom_api_key;
            setHasLLMChanges(hostChanged || keyChanged || customBaseUrlChanged || customModelChanged || customKeyChanged);
        }
    }, [ollamaPreset, customOllamaUrl, openrouterKey, customApiBaseUrl, customApiKey, customApiModel, llmSettings]);

    // Check if currently reloading
    const isCurrentlyReloading = startupStatus?.status === 'loading' || startupStatus?.status === 'downloading';

    const handleSave = async () => {
        setIsSaving(true);
        try {
            await onSave(settings);
            setSaveSuccess(true);
            setTimeout(() => setSaveSuccess(false), 2000);
        } finally {
            setIsSaving(false);
        }
    };

    const handleSaveLLM = async () => {
        setIsSavingLLM(true);
        try {
            const updates: {
                ollama_host?: string;
                openrouter_api_key?: string;
                custom_api_base_url?: string;
                custom_api_key?: string;
                custom_api_model?: string;
            } = {};

            // Get effective URL from preset or custom
            const effectiveUrl = ollamaPreset === 'custom' ? customOllamaUrl : OLLAMA_PRESETS[ollamaPreset as keyof typeof OLLAMA_PRESETS].url;
            if (effectiveUrl !== (llmSettings?.ollama_host || '')) {
                updates.ollama_host = effectiveUrl;
            }
            if (openrouterKey && openrouterKey !== llmSettings?.openrouter_api_key) {
                updates.openrouter_api_key = openrouterKey;
            }
            // Custom API updates
            if (customApiBaseUrl !== (llmSettings?.custom_api_base_url || '')) {
                updates.custom_api_base_url = customApiBaseUrl;
            }
            if (customApiModel !== (llmSettings?.custom_api_model || '')) {
                updates.custom_api_model = customApiModel;
            }
            if (customApiKey && customApiKey !== llmSettings?.custom_api_key) {
                updates.custom_api_key = customApiKey;
            }
            await onSaveLLM(updates);
            setSaveLLMSuccess(true);
            setHasLLMChanges(false);
            setTimeout(() => setSaveLLMSuccess(false), 2000);
        } finally {
            setIsSavingLLM(false);
        }
    };

    const handleReload = async () => {
        setIsReloading(true);
        try {
            await onReload(settings);
            // Don't close modal - let user see reload progress
        } catch (error: any) {
            alert(error.message || 'Failed to reload models');
            setIsReloading(false);
        }
    };

    // Reset isReloading when reload completes
    useEffect(() => {
        if (startupStatus?.status === 'ready' && isReloading) {
            setIsReloading(false);
        }
    }, [startupStatus?.status, isReloading]);

    if (!isOpen) return null;

    const selectClass = `w-full px-3 py-2 rounded-lg border transition-colors ${
        darkMode
            ? 'bg-[#282828] border-[#383838] text-white focus:border-[#1DB954]'
            : 'bg-white border-slate-300 text-slate-900 focus:border-cyan-500'
    } focus:outline-none focus:ring-2 focus:ring-opacity-30 ${
        darkMode ? 'focus:ring-[#1DB954]' : 'focus:ring-cyan-500'
    }`;

    const labelClass = `block text-sm font-medium mb-1 ${
        darkMode ? 'text-[#b3b3b3]' : 'text-slate-600'
    }`;

    return (
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                        onClick={onClose}
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className={`relative w-full max-w-lg rounded-xl shadow-2xl overflow-hidden ${
                            darkMode ? 'bg-[#181818]' : 'bg-white'
                        }`}
                    >
                        {/* Header */}
                        <div className={`flex items-center justify-between px-6 py-4 border-b ${
                            darkMode ? 'border-[#282828]' : 'border-slate-200'
                        }`}>
                            <div className="flex items-center gap-3">
                                <Settings2 className={`w-5 h-5 ${darkMode ? 'text-[#1DB954]' : 'text-cyan-500'}`} />
                                <h2 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-slate-900'}`}>
                                    Settings
                                </h2>
                            </div>
                            <button
                                onClick={onClose}
                                className={`p-2 rounded-full transition-colors ${
                                    darkMode ? 'hover:bg-[#282828] text-[#b3b3b3]' : 'hover:bg-slate-100 text-slate-500'
                                }`}
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="px-6 py-4 max-h-[70vh] overflow-y-auto">
                            {/* GPU Hardware Section */}
                            <div className="mb-6">
                                <h3 className={`flex items-center gap-2 text-sm font-semibold uppercase tracking-wide mb-3 ${
                                    darkMode ? 'text-white' : 'text-slate-900'
                                }`}>
                                    <Cpu className="w-4 h-4" />
                                    GPU Hardware
                                </h3>
                                <div className={`p-4 rounded-lg ${
                                    darkMode ? 'bg-[#282828]' : 'bg-slate-50'
                                }`}>
                                    {!gpuStatus?.cuda_available ? (
                                        <p className={darkMode ? 'text-[#b3b3b3]' : 'text-slate-600'}>
                                            No CUDA GPU detected
                                        </p>
                                    ) : (
                                        <div className="space-y-2">
                                            {gpuStatus.gpus.map((gpu) => (
                                                <div key={gpu.index} className="flex items-center justify-between">
                                                    <div className="flex items-center gap-2">
                                                        <span className={`font-medium ${
                                                            darkMode ? 'text-white' : 'text-slate-900'
                                                        }`}>
                                                            GPU {gpu.index}:
                                                        </span>
                                                        <span className={darkMode ? 'text-[#b3b3b3]' : 'text-slate-600'}>
                                                            {gpu.name}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <span className={`text-sm px-2 py-0.5 rounded ${
                                                            darkMode ? 'bg-[#383838] text-[#b3b3b3]' : 'bg-slate-200 text-slate-600'
                                                        }`}>
                                                            {gpu.vram_gb} GB
                                                        </span>
                                                        {gpu.supports_flash_attention && (
                                                            <span className={`text-xs px-2 py-0.5 rounded ${
                                                                darkMode ? 'bg-[#1DB954]/20 text-[#1DB954]' : 'bg-cyan-100 text-cyan-700'
                                                            }`}>
                                                                Flash OK
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Configuration Section */}
                            <div className="mb-6">
                                <h3 className={`text-sm font-semibold uppercase tracking-wide mb-3 ${
                                    darkMode ? 'text-white' : 'text-slate-900'
                                }`}>
                                    Configuration
                                </h3>
                                <div className="space-y-4">
                                    {/* 4-bit Quantization */}
                                    <div>
                                        <label className={labelClass}>4-bit Quantization</label>
                                        <select
                                            value={settings.quantization_4bit}
                                            onChange={(e) => {
                                                const newValue = e.target.value;
                                                if (newValue !== 'false' && settings.torch_compile) {
                                                    // Enabling 4-bit - disable torch.compile
                                                    setSettings({
                                                        ...settings,
                                                        quantization_4bit: newValue,
                                                        torch_compile: false
                                                    });
                                                } else {
                                                    setSettings({ ...settings, quantization_4bit: newValue });
                                                }
                                            }}
                                            className={selectClass}
                                        >
                                            <option value="auto">Auto (based on VRAM)</option>
                                            <option value="true">Enabled</option>
                                            <option value="false">Disabled (required for torch.compile)</option>
                                        </select>
                                        <p className={`text-xs mt-1 ${darkMode ? 'text-[#6a6a6a]' : 'text-slate-400'}`}>
                                            Reduces VRAM usage from ~11GB to ~3GB
                                        </p>
                                    </div>

                                    {/* Sequential Offload / Memory Swap Mode */}
                                    <div>
                                        <label className={labelClass}>Memory Swap Mode</label>
                                        <select
                                            value={settings.sequential_offload}
                                            onChange={(e) => setSettings({ ...settings, sequential_offload: e.target.value })}
                                            className={selectClass}
                                        >
                                            <option value="auto">Auto (based on VRAM)</option>
                                            <option value="true">Force Enabled (recommended if getting CUBLAS errors)</option>
                                            <option value="false">Disabled</option>
                                        </select>
                                        <p className={`text-xs mt-1 ${darkMode ? 'text-[#6a6a6a]' : 'text-slate-400'}`}>
                                            Loads one model at a time to reduce VRAM usage. Enable if you see CUBLAS_STATUS_EXECUTION_FAILED errors.
                                        </p>
                                    </div>

                                    {/* torch.compile */}
                                    <div>
                                        <div className="flex items-center justify-between">
                                            <label className={labelClass}>torch.compile</label>
                                            <button
                                                onClick={() => {
                                                    const newCompile = !settings.torch_compile;
                                                    if (newCompile) {
                                                        // Enabling torch.compile - disable 4-bit quantization
                                                        setSettings({
                                                            ...settings,
                                                            torch_compile: true,
                                                            quantization_4bit: 'false'
                                                        });
                                                    } else {
                                                        setSettings({ ...settings, torch_compile: false });
                                                    }
                                                }}
                                                className={`relative w-11 h-6 rounded-full transition-colors ${
                                                    settings.torch_compile
                                                        ? darkMode ? 'bg-[#1DB954]' : 'bg-cyan-500'
                                                        : darkMode ? 'bg-[#383838]' : 'bg-slate-300'
                                                }`}
                                            >
                                                <span className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                                                    settings.torch_compile ? 'translate-x-5' : ''
                                                }`} />
                                            </button>
                                        </div>
                                        <p className={`text-xs mt-1 ${darkMode ? 'text-[#6a6a6a]' : 'text-slate-400'}`}>
                                            ~2x faster inference (requires full precision)
                                        </p>
                                        {settings.torch_compile && (
                                            <>
                                                {/* Warning for older GPUs */}
                                                {gpuStatus?.gpus && Object.values(gpuStatus.gpus).some(
                                                    (gpu: { compute_capability?: number }) => gpu.compute_capability && gpu.compute_capability < 7.5
                                                ) && (
                                                    <div className={`mt-2 p-2 rounded text-xs ${
                                                        darkMode ? 'bg-amber-900/30 text-amber-400' : 'bg-amber-50 text-amber-700'
                                                    }`}>
                                                        <strong>⚠ Warning:</strong> Your GPU (SM {
                                                            Object.values(gpuStatus.gpus).find(
                                                                (gpu: { compute_capability?: number }) => gpu.compute_capability && gpu.compute_capability < 7.5
                                                            )?.compute_capability
                                                        }) is older than recommended for torch.compile.
                                                        torch.compile works best on Turing (SM 7.5+) or newer GPUs (RTX 20xx/30xx/40xx, A100, etc.).
                                                        On older GPUs, compilation may be very slow or fail. The backend will auto-disable it for stability.
                                                    </div>
                                                )}
                                                <div className={`mt-2 p-2 rounded text-xs ${
                                                    darkMode ? 'bg-blue-900/20 text-blue-400' : 'bg-blue-50 text-blue-700'
                                                }`}>
                                                    <strong>Note:</strong> 4-bit quantization has been disabled for torch.compile compatibility.
                                                    First generation will take 5-10 minutes to compile. Subsequent runs will be ~2x faster.
                                                    Requires ~11GB VRAM without quantization.
                                                </div>
                                            </>
                                        )}
                                    </div>

                                    {/* torch.compile mode */}
                                    {settings.torch_compile && (
                                        <div>
                                            <label className={labelClass}>Compile Mode</label>
                                            <select
                                                value={settings.torch_compile_mode}
                                                onChange={(e) => setSettings({ ...settings, torch_compile_mode: e.target.value })}
                                                className={selectClass}
                                            >
                                                <option value="default">Default</option>
                                                <option value="reduce-overhead">Reduce Overhead</option>
                                                <option value="max-autotune">Max Autotune</option>
                                            </select>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* LLM Provider Section */}
                            <div className="mb-6">
                                <h3 className={`flex items-center gap-2 text-sm font-semibold uppercase tracking-wide mb-3 ${
                                    darkMode ? 'text-white' : 'text-slate-900'
                                }`}>
                                    <Globe className="w-4 h-4" />
                                    LLM Providers
                                </h3>
                                <div className="space-y-4">
                                    {/* Ollama Location */}
                                    <div>
                                        <label className={labelClass}>
                                            <div className="flex items-center gap-2">
                                                Ollama Location
                                                {llmSettings?.ollama_available ? (
                                                    <CheckCircle className="w-3.5 h-3.5 text-green-500" />
                                                ) : (
                                                    <XCircle className="w-3.5 h-3.5 text-red-400" />
                                                )}
                                            </div>
                                        </label>
                                        {/* Preset buttons */}
                                        <div className="flex flex-wrap gap-2 mb-2">
                                            {Object.entries(OLLAMA_PRESETS).map(([key, preset]) => (
                                                <button
                                                    key={key}
                                                    onClick={() => setOllamaPreset(key)}
                                                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                                                        ollamaPreset === key
                                                            ? darkMode
                                                                ? 'bg-[#1DB954] text-black'
                                                                : 'bg-cyan-500 text-white'
                                                            : darkMode
                                                                ? 'bg-[#282828] text-[#b3b3b3] hover:bg-[#383838]'
                                                                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                                    }`}
                                                >
                                                    {preset.label}
                                                </button>
                                            ))}
                                        </div>
                                        {/* Description for selected preset */}
                                        <p className={`text-xs mb-2 ${darkMode ? 'text-[#b3b3b3]' : 'text-slate-500'}`}>
                                            {OLLAMA_PRESETS[ollamaPreset as keyof typeof OLLAMA_PRESETS].desc}
                                        </p>
                                        {/* Custom URL input - only show when custom is selected */}
                                        {ollamaPreset === 'custom' && (
                                            <input
                                                type="text"
                                                value={customOllamaUrl}
                                                onChange={(e) => setCustomOllamaUrl(e.target.value)}
                                                placeholder="http://192.168.1.100:11434"
                                                className={selectClass}
                                            />
                                        )}
                                        {/* Show current URL for non-custom presets */}
                                        {ollamaPreset !== 'custom' && (
                                            <div className={`text-xs px-3 py-2 rounded ${
                                                darkMode ? 'bg-[#282828] text-[#6a6a6a]' : 'bg-slate-50 text-slate-400'
                                            }`}>
                                                {OLLAMA_PRESETS[ollamaPreset as keyof typeof OLLAMA_PRESETS].url}
                                            </div>
                                        )}
                                    </div>

                                    {/* OpenRouter API Key */}
                                    <div>
                                        <label className={labelClass}>
                                            <div className="flex items-center gap-2">
                                                <Key className="w-3.5 h-3.5" />
                                                OpenRouter API Key
                                                {llmSettings?.openrouter_available ? (
                                                    <CheckCircle className="w-3.5 h-3.5 text-green-500" />
                                                ) : (
                                                    <span className={`text-xs ${darkMode ? 'text-[#6a6a6a]' : 'text-slate-400'}`}>(not set)</span>
                                                )}
                                            </div>
                                        </label>
                                        <input
                                            type="password"
                                            value={openrouterKey}
                                            onChange={(e) => setOpenrouterKey(e.target.value)}
                                            placeholder={llmSettings?.openrouter_api_key || 'sk-or-...'}
                                            className={selectClass}
                                        />
                                        <p className={`text-xs mt-1 ${darkMode ? 'text-[#6a6a6a]' : 'text-slate-400'}`}>
                                            Get your API key from <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer" className={darkMode ? 'text-[#1DB954] hover:underline' : 'text-cyan-500 hover:underline'}>openrouter.ai/keys</a>
                                        </p>
                                    </div>

                                    {/* Custom API Section */}
                                    <div className={`mt-4 pt-4 border-t ${darkMode ? 'border-[#383838]' : 'border-slate-200'}`}>
                                        <label className={`${labelClass} mb-2`}>
                                            <div className="flex items-center gap-2">
                                                Custom API (OpenAI Compatible)
                                                {llmSettings?.custom_api_available ? (
                                                    <CheckCircle className="w-3.5 h-3.5 text-green-500" />
                                                ) : (
                                                    <span className={`text-xs ${darkMode ? 'text-[#6a6a6a]' : 'text-slate-400'}`}>(not configured)</span>
                                                )}
                                            </div>
                                        </label>
                                        <p className={`text-xs mb-3 ${darkMode ? 'text-[#b3b3b3]' : 'text-slate-500'}`}>
                                            Connect to vLLM, LocalAI, LM Studio, or any OpenAI-compatible endpoint
                                        </p>

                                        {/* Base URL */}
                                        <div className="mb-3">
                                            <label className={`text-xs ${darkMode ? 'text-[#b3b3b3]' : 'text-slate-500'}`}>Base URL</label>
                                            <input
                                                type="text"
                                                value={customApiBaseUrl}
                                                onChange={(e) => setCustomApiBaseUrl(e.target.value)}
                                                placeholder="http://localhost:8080/v1"
                                                className={selectClass}
                                            />
                                        </div>

                                        {/* Model Name */}
                                        <div className="mb-3">
                                            <label className={`text-xs ${darkMode ? 'text-[#b3b3b3]' : 'text-slate-500'}`}>Model Name</label>
                                            <input
                                                type="text"
                                                value={customApiModel}
                                                onChange={(e) => setCustomApiModel(e.target.value)}
                                                placeholder="llama-3.1-8b"
                                                className={selectClass}
                                            />
                                        </div>

                                        {/* API Key (Optional) */}
                                        <div>
                                            <label className={`text-xs ${darkMode ? 'text-[#b3b3b3]' : 'text-slate-500'}`}>API Key (optional)</label>
                                            <input
                                                type="password"
                                                value={customApiKey}
                                                onChange={(e) => setCustomApiKey(e.target.value)}
                                                placeholder={llmSettings?.custom_api_key || 'Leave empty if not required'}
                                                className={selectClass}
                                            />
                                            <p className={`text-xs mt-1 ${darkMode ? 'text-[#6a6a6a]' : 'text-slate-400'}`}>
                                                Only needed if your API requires authentication
                                            </p>
                                        </div>
                                    </div>

                                    {/* LLM Save Button */}
                                    <div className="flex flex-col gap-2">
                                        <div className="flex items-center gap-3">
                                            <button
                                                onClick={handleSaveLLM}
                                                disabled={isSavingLLM || !hasLLMChanges}
                                                className={`px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                                                    darkMode
                                                        ? 'bg-[#1DB954] text-black hover:bg-[#1ed760]'
                                                        : 'bg-cyan-500 text-white hover:bg-cyan-600'
                                                }`}
                                            >
                                                {isSavingLLM ? 'Saving...' : 'Save LLM Settings'}
                                            </button>
                                            {saveLLMSuccess && (
                                                <span className={`flex items-center gap-1 text-sm ${
                                                    darkMode ? 'text-[#1DB954]' : 'text-cyan-600'
                                                }`}>
                                                    <Check className="w-4 h-4" />
                                                    Saved
                                                </span>
                                            )}
                                        </div>
                                        <p className={`text-xs ${darkMode ? 'text-[#6a6a6a]' : 'text-slate-400'}`}>
                                            LLM settings take effect immediately - no model reload needed
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Warning */}
                            {hasChanges && (
                                <div className={`flex items-start gap-3 p-4 rounded-lg mb-4 ${
                                    darkMode ? 'bg-amber-900/20 border border-amber-800' : 'bg-amber-50 border border-amber-200'
                                }`}>
                                    <AlertTriangle className={`w-5 h-5 flex-shrink-0 ${
                                        darkMode ? 'text-amber-400' : 'text-amber-600'
                                    }`} />
                                    <div>
                                        <p className={`text-sm font-medium ${
                                            darkMode ? 'text-amber-400' : 'text-amber-800'
                                        }`}>
                                            Changes require model reload
                                        </p>
                                        <p className={`text-xs mt-1 ${
                                            darkMode ? 'text-amber-400/70' : 'text-amber-700'
                                        }`}>
                                            This will take 1-3 minutes. No generation during reload.
                                        </p>
                                    </div>
                                </div>
                            )}

                            {/* Reload Progress */}
                            {isCurrentlyReloading && (
                                <div className={`p-4 rounded-lg mb-4 ${
                                    darkMode ? 'bg-[#282828]' : 'bg-slate-100'
                                }`}>
                                    <div className="flex items-center gap-3 mb-2">
                                        <RefreshCw className={`w-4 h-4 animate-spin ${
                                            darkMode ? 'text-[#1DB954]' : 'text-cyan-500'
                                        }`} />
                                        <span className={`text-sm font-medium ${
                                            darkMode ? 'text-white' : 'text-slate-900'
                                        }`}>
                                            {startupStatus?.message || 'Reloading...'}
                                        </span>
                                    </div>
                                    <div className={`w-full h-2 rounded-full overflow-hidden ${
                                        darkMode ? 'bg-[#383838]' : 'bg-slate-200'
                                    }`}>
                                        <div
                                            className={`h-full rounded-full transition-all duration-300 ${
                                                darkMode ? 'bg-[#1DB954]' : 'bg-cyan-500'
                                            }`}
                                            style={{ width: `${startupStatus?.progress || 0}%` }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Footer */}
                        <div className={`flex items-center justify-end gap-3 px-6 py-4 border-t ${
                            darkMode ? 'border-[#282828]' : 'border-slate-200'
                        }`}>
                            {saveSuccess && (
                                <span className={`flex items-center gap-1 text-sm ${
                                    darkMode ? 'text-[#1DB954]' : 'text-cyan-600'
                                }`}>
                                    <Check className="w-4 h-4" />
                                    Saved
                                </span>
                            )}
                            <button
                                onClick={handleSave}
                                disabled={isSaving || !hasChanges || isCurrentlyReloading}
                                className={`px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                                    darkMode
                                        ? 'bg-[#282828] text-white hover:bg-[#383838]'
                                        : 'bg-slate-100 text-slate-900 hover:bg-slate-200'
                                }`}
                            >
                                {isSaving ? 'Saving...' : 'Save'}
                            </button>
                            <button
                                onClick={handleReload}
                                disabled={isReloading || isCurrentlyReloading}
                                className={`px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 ${
                                    darkMode
                                        ? 'bg-[#1DB954] text-black hover:bg-[#1ed760]'
                                        : 'bg-cyan-500 text-white hover:bg-cyan-600'
                                }`}
                            >
                                {(isReloading || isCurrentlyReloading) ? (
                                    <>
                                        <RefreshCw className="w-4 h-4 animate-spin" />
                                        Reloading...
                                    </>
                                ) : (
                                    <>
                                        <RefreshCw className="w-4 h-4" />
                                        Apply & Reload
                                    </>
                                )}
                            </button>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
}
