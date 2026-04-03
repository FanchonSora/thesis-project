/**
 * Brain Tumor Analysis Platform - Main Application
 */

class BrainAnalysisApp {
    constructor() {
        this.uploadedFiles = {};
        this.currentJobId = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.mesh = null;
        this.statusStartTime = null;
        this.pollTimer = null;
        this.init();
    }

    /* =====================================
       INITIALIZATION
       ===================================== */

    init() {
        this.setupEventListeners();
        this.handleDragDrop();
    }

    setupEventListeners() {
        // Case ID input
        document.getElementById('caseId').addEventListener('change', () => {
            this.updateSubmitButtonState();
        });

        // File upload
        document.querySelectorAll('.file-input').forEach(input => {
            input.addEventListener('change', (e) => this.handleFileSelect(e));
        });

        // Drop zones
        document.querySelectorAll('.drop-zone').forEach(zone => {
            zone.addEventListener('click', (e) => {
                const modality = e.currentTarget.getAttribute('data-modality');
                const input = document.querySelector(`input[data-modality="${modality}"]`);
                input.click();
            });
        });

        // Buttons
        document.getElementById('submitBtn').addEventListener('click', () => this.submitAnalysis());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearAll());

        // Report tabs
        document.querySelectorAll('.report-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.getAttribute('data-tab')));
        });

        // Viewer controls
        document.getElementById('controlReset').addEventListener('click', () => this.resetView());
        document.getElementById('controlDownload').addEventListener('click', () => this.downloadMesh());

        // Download buttons
        document.getElementById('downloadReport').addEventListener('click', () => {
            this.downloadFile('report');
        });
        document.getElementById('downloadPrediction').addEventListener('click', () => {
            this.downloadFile('pred_post');
        });
        document.getElementById('downloadMesh').addEventListener('click', () => {
            this.downloadFile('mesh');
        });
    }

    handleDragDrop() {
        document.querySelectorAll('.drop-zone').forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.add('dragover');
            });

            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.remove('dragover');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.remove('dragover');
                
                const modality = zone.getAttribute('data-modality');
                const files = e.dataTransfer.files;
                
                if (files.length > 0) {
                    const input = document.querySelector(`input[data-modality="${modality}"]`);
                    input.files = files;
                    this.handleFileSelect({ target: input });
                }
            });
        });
    }

    stopPolling() {
        if (this.pollTimer) {
            clearTimeout(this.pollTimer);
            this.pollTimer = null;
        }
    }

    /* =====================================
       FILE HANDLING
       ===================================== */

    handleFileSelect(event) {
        const input = event.target;
        const modality = input.getAttribute('data-modality');
        const file = input.files[0];

        if (!file) return;

        // Validate file
        if (!file.name.endsWith('.nii') && !file.name.endsWith('.nii.gz')) {
            this.showToast('Invalid file format. Please upload .nii or .nii.gz files.', 'error');
            return;
        }

        const maxSize = 500 * 1024 * 1024; // 500MB
        if (file.size > maxSize) {
            this.showToast(`File too large. Maximum size is 500MB.`, 'error');
            return;
        }

        // Store file
        this.uploadedFiles[modality] = {
            file: file,
            name: file.name,
            size: file.size
        };

        // Update UI
        this.updateModality(modality);
        this.updateSubmitButtonState();
        this.showToast(`${modality.toUpperCase()} uploaded successfully`, 'success');
    }

    updateModality(modality) {
        const card = document.querySelector(`[data-modality="${modality}"]`);
        const dropZone = card.querySelector('.drop-zone');
        const fileInfo = card.querySelector('.file-info');
        const file = this.uploadedFiles[modality];

        card.classList.add('active');
        dropZone.style.display = 'none';
        fileInfo.style.display = 'block';

        const sizeStr = this.formatBytes(file.size);
        fileInfo.innerHTML = `
            <small class="file-name">${file.name}</small>
            <small class="file-size">${sizeStr}</small>
        `;
    }

    clearAll() {
        this.stopPolling();
        this.currentJobId = null;

        const resultsContainer = document.getElementById('resultsContainer');
        const emptyState = document.getElementById('emptyState');
        const statusCard = document.getElementById('statusCard');

        this.uploadedFiles = {};

        document.querySelectorAll('.modality-card').forEach(card => {
            const modality = card.getAttribute('data-modality');
            const input = document.querySelector(`input[data-modality="${modality}"]`);
            const dropZone = card.querySelector('.drop-zone');
            const fileInfo = card.querySelector('.file-info');

            input.value = '';
            card.classList.remove('active');
            dropZone.style.display = 'flex';
            fileInfo.style.display = 'none';
        });

        document.getElementById('caseId').value = '';
        resultsContainer.style.display = 'none';
        statusCard.style.display = 'none';
        emptyState.style.display = 'flex';

        this.updateSubmitButtonState();
        this.showToast('All uploads cleared', 'success');
    }

    /* =====================================
       SUBMISSION & ANALYSIS
       ===================================== */

    updateSubmitButtonState() {
        const caseId = document.getElementById('caseId').value.trim();
        const fileCount = Object.keys(this.uploadedFiles).length;
        const submitBtn = document.getElementById('submitBtn');
        
        const isValid = caseId && fileCount >= 3;
        submitBtn.disabled = !isValid;
    }

    async submitAnalysis() {
        const caseId = document.getElementById('caseId').value.trim();
        const fileCount = Object.keys(this.uploadedFiles).length;

        if (!caseId || fileCount < 3) {
            this.showToast('Please enter case ID and upload at least 3 modalities', 'error');
            return;
        }

        // Stop old polling before creating a new job
        this.stopPolling();
        this.currentJobId = null;

        const formData = new FormData();
        formData.append('case_id', caseId);

        for (const [modality, fileObj] of Object.entries(this.uploadedFiles)) {
            formData.append(modality, fileObj.file);
        }

        this.showStatusCard();
        this.statusStartTime = Date.now();

        try {
            const response = await fetch('/jobs', {
                method: 'POST',
                body: formData
            });
            const raw = await response.text();
            let data = {};
            try {
                data = raw ? JSON.parse(raw) : {};
            } catch {
                throw new Error('Invalid response from /jobs');
            }
            if (!response.ok) {
                this.showToast(data.detail || 'Failed to submit analysis', 'error');
                this.updateStatus(data.detail || 'Failed to submit analysis', 'failed');
                return;
            }

            this.currentJobId = data.job_id;
            this.updateStatus('Job submitted. Waiting to start...', 'queued');
            this.pollJobStatus();
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
            this.updateStatus(`Error: ${error.message}`, 'failed');
        }
    }

    async pollJobStatus() {
        if (!this.currentJobId) return;

        const jobIdAtRequestTime = this.currentJobId;

        try {
            const response = await fetch(`/jobs/${jobIdAtRequestTime}/status`, {
                cache: 'no-store',
                headers: {
                    'Accept': 'application/json'
                }
            });

            if (response.status === 404) {
                if (this.currentJobId === jobIdAtRequestTime) {
                    this.stopPolling();
                    this.currentJobId = null;
                    this.updateStatus('Job not found. Server may have restarted. Please run again.', 'failed');
                    this.showToast('Job not found. Server may have restarted.', 'error');
                }
                return;
            }

            const raw = await response.text();

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${raw || 'Unknown server error'}`);
            }

            if (!raw || !raw.trim()) {
                throw new Error('Empty status response');
            }

            let job;
            try {
                job = JSON.parse(raw);
            } catch (e) {
                console.error('Invalid status JSON:', raw);
                throw new Error(`Invalid JSON from status endpoint: ${e.message}`);
            }

            if (!job || typeof job !== 'object') {
                throw new Error('Invalid job payload');
            }

            if (this.currentJobId !== jobIdAtRequestTime) return;

            const status = job.status || 'queued';
            this.updateStatus(`Status: ${status}...`, status);

            if (status === 'running' || status === 'queued') {
                this.stopPolling();
                this.pollTimer = setTimeout(() => this.pollJobStatus(), 2000);
                return;
            }

            this.stopPolling();

            if (status === 'completed') {
                try {
                    await this.showResults();
                    this.updateStatus('Analysis completed.', 'completed');
                    this.showToast('Analysis completed successfully!', 'success');
                } catch (e) {
                    console.error('showResults failed:', e);
                    this.updateStatus(`Completed but failed to render results: ${e.message}`, 'failed');
                    this.showToast(`Render failed: ${e.message}`, 'error');
                }
                return;
            }

            if (status === 'failed') {
                const errorMsg = job.error || 'Unknown error';
                this.updateStatus(`Failed: ${errorMsg}`, 'failed');
                this.showToast(`Analysis failed: ${errorMsg}`, 'error');
                return;
            }

            this.updateStatus(`Unknown status: ${status}`, 'failed');
            this.showToast(`Unknown job status: ${status}`, 'error');

        } catch (error) {
            console.error('Poll error:', error);
            this.updateStatus(`Polling error: ${error.message}`, 'failed');
            this.stopPolling();
            this.pollTimer = setTimeout(() => this.pollJobStatus(), 5000);
        }
    }

    /* =====================================
       STATUS & UI UPDATES
       ===================================== */

    showStatusCard() {
        const statusCard = document.getElementById('statusCard');
        const emptyState = document.getElementById('emptyState');
        
        statusCard.style.display = 'block';
        emptyState.style.display = 'none';
    }

    updateStatus(message, status) {
        const elapsedSeconds = Math.floor((Date.now() - this.statusStartTime) / 1000);
        const minutes = Math.floor(elapsedSeconds / 60);
        const seconds = elapsedSeconds % 60;
        const timeStr = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

        document.getElementById('statusTitle').textContent = status.charAt(0).toUpperCase() + status.slice(1);
        document.getElementById('statusMessage').textContent = message;
        document.getElementById('statusTime').textContent = timeStr;

        // Update progress
        const progressMap = {
            'queued': 10,
            'running': 50,
            'completed': 100,
            'failed': 100
        };
        
        const width = progressMap[status] || 10;
        document.getElementById('progressFill').style.width = width + '%';
    }

    async showResults() {
        const statusCard = document.getElementById('statusCard');
        const resultsContainer = document.getElementById('resultsContainer');
        const emptyState = document.getElementById('emptyState');

        // Chỉ đổi UI sau khi load thật sự thành công
        await this.loadResults();
        await this.initalize3DViewer();

        statusCard.style.display = 'none';
        emptyState.style.display = 'none';
        resultsContainer.style.display = 'flex';
    }

    async loadResults() {
        const response = await fetch(`/jobs/${this.currentJobId}/report`, {
            cache: 'no-store',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        const report = await response.json();

        document.getElementById('metricStatus').textContent = '✓ ' + (report.status || 'Complete');

        const timeSeconds = Math.floor((Date.now() - this.statusStartTime) / 1000);
        const minutes = Math.floor(timeSeconds / 60);
        const seconds = timeSeconds % 60;
        document.getElementById('metricTime').textContent = `${minutes}m ${seconds}s`;

        document.getElementById('metricModel').textContent = 'UNet3D';
        document.getElementById('metricSynthesis').textContent =
            report.synthesis_status && report.synthesis_status !== 'skipped' ? 'On' : 'Off';

        this.updateDetailsTab(report);
        this.updateVolumesTab(report);

        return report;
    }

    updateDetailsTab(report) {
        const detailsContent = document.getElementById('detailsContent');
        let html = '<div class="details-list">';

        html += `<div class="detail-item">
            <span class="detail-label">Case ID:</span>
            <span class="detail-value">${report.case_id}</span>
        </div>`;

        html += `<div class="detail-item">
            <span class="detail-label">Status:</span>
            <span class="detail-value">${report.status}</span>
        </div>`;

        if (report.missing_flags) {
            const missing = Object.entries(report.missing_flags)
                .filter(([_, v]) => v === 1)
                .map(([k]) => k.toUpperCase())
                .join(', ') || 'None';
            
            html += `<div class="detail-item">
                <span class="detail-label">Missing Modalities:</span>
                <span class="detail-value">${missing}</span>
            </div>`;
        }

        if (report.synthesis_status) {
            html += `<div class="detail-item">
                <span class="detail-label">Synthesis:</span>
                <span class="detail-value">${report.synthesis_status}</span>
            </div>`;
        }

        if (report.downsample_factor) {
            html += `<div class="detail-item">
                <span class="detail-label">Downsample Factor:</span>
                <span class="detail-value">${report.downsample_factor.toFixed(2)}x</span>
            </div>`;
        }

        html += '</div>';
        detailsContent.innerHTML = html;
    }

    updateVolumesTab(report) {
        const volumesChart = document.getElementById('volumesChart');
        
        if (!report.region_volumes_mm3) {
            volumesChart.innerHTML = '<p>No volume data available</p>';
            return;
        }

        let html = '<div class="volumes-list">';
        for (const [region, volume] of Object.entries(report.region_volumes_mm3)) {
            const volumeCm3 = (volume / 1000).toFixed(2);
            html += `<div class="volume-item">
                <span class="volume-label">${region}</span>
                <span class="volume-value">${volumeCm3} cm³</span>
            </div>`;
        }
        html += '</div>';
        
        volumesChart.innerHTML = html;
    }

    switchTab(tabName) {
        // Update active tabs
        document.querySelectorAll('.report-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`.report-tab[data-tab="${tabName}"]`).classList.add('active');

        // Update active content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName + 'Tab').classList.add('active');
    }

    /* =====================================
       3D VISUALIZATION
       ===================================== */

    async initalize3DViewer() {
        const container = document.getElementById('threejsContainer');
        if (!container) {
            throw new Error('3D viewer container not found');
        }

        // clear viewer cũ nếu có
        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }
        container.innerHTML = '';

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1e293b);

        this.camera = new THREE.PerspectiveCamera(
            75,
            container.clientWidth / container.clientHeight,
            0.1,
            10000
        );
        this.camera.position.z = 200;

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 2;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(100, 100, 100);
        this.scene.add(directionalLight);

        await this.loadMesh();

        const animate = () => {
            if (!this.renderer || !this.scene || !this.camera) return;
            requestAnimationFrame(animate);
            this.controls?.update();
            this.renderer.render(this.scene, this.camera);
        };
        animate();

        window.addEventListener('resize', () => this.onWindowResize());
    }

    async loadMesh() {
        try {
            const response = await fetch(`/jobs/${this.currentJobId}/file/mesh`, {
                cache: 'no-store'
            });

            if (!response.ok) {
                throw new Error(`Mesh file not available (HTTP ${response.status})`);
            }

            // Tạm thời chưa parse OBJ thật, nên vẫn dùng placeholder
            await response.blob();

            const geometry = new THREE.BoxGeometry(100, 100, 100);
            const material = new THREE.MeshPhongMaterial({
                color: 0x3b82f6,
                emissive: 0x1e40af,
                shininess: 100,
                wireframe: false
            });

            this.mesh = new THREE.Mesh(geometry, material);
            this.scene.add(this.mesh);

            const bbox = new THREE.Box3().setFromObject(this.mesh);
            const center = bbox.getCenter(new THREE.Vector3());
            this.camera.position.copy(center);
            this.camera.position.z += bbox.getSize(new THREE.Vector3()).z * 1.5;
            this.controls.target.copy(center);
            this.controls.update();

        } catch (error) {
            console.error('Error loading mesh:', error);
            this.createPlaceholderGeometry();
        }
    }

    createPlaceholderGeometry() {
        // Create some geometric shapes for visualization
        const group = new THREE.Group();

        // ET (Enhancing Tumor) - Red
        const etGeom = new THREE.SphereGeometry(30, 32, 32);
        const etMat = new THREE.MeshPhongMaterial({ color: 0xef4444 });
        const etMesh = new THREE.Mesh(etGeom, etMat);
        etMesh.position.z = -20;
        group.add(etMesh);

        // TC (Tumor Core) - Purple
        const tcGeom = new THREE.CylinderGeometry(40, 40, 60, 32);
        const tcMat = new THREE.MeshPhongMaterial({ color: 0xa78bfa });
        const tcMesh = new THREE.Mesh(tcGeom, tcMat);
        tcMesh.position.z = 0;
        group.add(tcMesh);

        // WT (Whole Tumor) - Teal (outline)
        const wtGeom = new THREE.BoxGeometry(100, 100, 100);
        const wtMat = new THREE.MeshPhongMaterial({ 
            color: 0x34d399,
            emissive: 0x0f3d2d,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        const wtMesh = new THREE.Mesh(wtGeom, wtMat);
        group.add(wtMesh);

        this.scene.add(group);
        this.mesh = group;
    }

    resetView() {
        if (this.mesh) {
            const bbox = new THREE.Box3().setFromObject(this.mesh);
            const center = bbox.getCenter(new THREE.Vector3());
            const size = bbox.getSize(new THREE.Vector3());
            
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = this.camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            
            this.camera.position.copy(center);
            this.camera.position.z += cameraZ * 1.5;
            this.controls.target.copy(center);
            this.controls.update();
        }
    }

    downloadMesh() {
        if (this.currentJobId) {
            window.location.href = `/jobs/${this.currentJobId}/file/mesh`;
        }
    }

    onWindowResize() {
        const container = document.getElementById('threejsContainer');
        if (!container || !this.camera || !this.renderer) return;

        const width = container.clientWidth;
        const height = container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /* =====================================
       FILE DOWNLOADS
       ===================================== */

    downloadFile(kind) {
        if (this.currentJobId) {
            window.location.href = `/jobs/${this.currentJobId}/file/${kind}`;
        }
    }

    /* =====================================
       UTILITIES
       ===================================== */

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span>${type === 'success' ? '✓' : type === 'error' ? '✕' : '⚠'}</span>
            <span>${message}</span>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideInRight 300ms ease-out reverse';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new BrainAnalysisApp();
});
