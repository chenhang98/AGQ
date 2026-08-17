import { exec } from 'child_process';
import * as os from 'os';

const X64_PROCESS_NAME = 'language_server_windows_x64.exe';
const ARM64_PROCESS_NAME = 'language_server_windows_arm.exe';

/**
 * Returns true when the current Windows machine is ARM64.
 *
 * On ARM64 devices running an x64 Node.js runtime (x64 emulation),
 * os.arch() still reports 'x64', so the processor architecture
 * environment variables are checked as a fallback.
 */
export function isWindowsArm64(): boolean {
    if (process.platform !== 'win32') {
        return false;
    }
    if (os.arch() === 'arm64' || process.arch === 'arm64') {
        return true;
    }
    return (
        process.env.PROCESSOR_ARCHITECTURE === 'ARM64' &&
        process.env.PROCESSOR_ARCHITEW6432 === 'AMD64'
    );
}

/**
 * Name of the Antigravity language server executable for the current
 * system architecture (language_server_windows_arm.exe on Windows
 * ARM64, language_server_windows_x64.exe on x64).
 */
export function getAntigravityProcessName(): string {
    return isWindowsArm64() ? ARM64_PROCESS_NAME : X64_PROCESS_NAME;
}

export function buildProcessListCommand(processName: string): string {
    const filter = `name='${processName}'`;
    return `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \\"${filter}\\" | Select-Object ProcessId,CommandLine | ConvertTo-Json"`;
}

export function listAntigravityProcesses(callback: (error: Error | null, json: string) => void): void {
    const command = buildProcessListCommand(getAntigravityProcessName());
    exec(command, { maxBuffer: 10 * 1024 * 1024 }, (error, stdout) => {
        if (error) {
            callback(error, '[]');
            return;
        }
        // Get-CimInstance prints an empty string (not valid JSON) when
        // no process matches the filter; normalize that to an empty list.
        const trimmed = stdout.trim();
        callback(null, trimmed.length > 0 ? trimmed : '[]');
    });
}
