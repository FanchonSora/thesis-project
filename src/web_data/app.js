class BrainAnalysisApp {
    constructor() {
        this.uploadedFiles = {};
        this.currentJobId = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.mesh = null;
        this.regionGroup = null;
        this.brainMeshObject = null;
        this.gridBox = null;
        this.statusStartTime = null;
        this.pollTimer = null;
        this.reportData = null;
        this.animationFrameId = null;
        this.resizeHandler = null;
        this.currentMeshLod = 'low';
        this.threeLoaded = false;
        this.init();
    }

    /* =====================================
       INITIALIZATION
       ===================================== */

    init() {
        this.setupEventListeners();
        this.handleDragDrop();
        this.ensureOptionalUI();
        this.ensureThreeJSLoaded();
    }

    ensureThreeJSLoaded() {
        // Check if THREE is already loaded
        if (window.THREE) {
            this.threeLoaded = true;
            return Promise.resolve();
        }

        // Load THREE.js from CDN as fallback
        return this.loadThreeJSFromCDN();
    }
    loadThreeJSFromCDN() {
        return new Promise((resolve) => {
            // Try loading from local files first
            const script = document.createElement('script');
            script.src = '/js/three.min.js';
            script.onload = () => {
                console.log('THREE.js loaded from local');

                const orbitScript = document.createElement('script');
                orbitScript.src = '/js/OrbitControls.js';
                orbitScript.onload = () => {
                    console.log('OrbitControls loaded from local');

                    const objLoaderScript = document.createElement('script');
                    objLoaderScript.src = '/js/OBJLoader.js';
                    objLoaderScript.onload = () => {
                        console.log('OBJLoader loaded from local');
                        this.threeLoaded = true;
                        resolve();
                    };
                    objLoaderScript.onerror = () => {
                        console.warn('Failed to load OBJLoader from local, trying CDN');
                        this.loadOBJLoaderFromCDN(resolve);
                    };
                    document.head.appendChild(objLoaderScript);
                };
                orbitScript.onerror = () => {
                    console.warn('Failed to load OrbitControls from local, trying CDN');
                    this.loadOrbitControlsFromCDN(resolve);
                };
                document.head.appendChild(orbitScript);
            };
            script.onerror = () => {
                console.warn('Failed to load THREE.js from local, trying CDN');
                this.loadFromCDN(resolve);
            };
            document.head.appendChild(script);
        });
    }

    loadFromCDN(resolve) {
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
        script.onload = () => {
            console.log('THREE.js loaded from CDN');
            this.loadOrbitControlsFromCDN(resolve);
        };
        script.onerror = () => {
            console.warn('Failed to load THREE.js from CDN');
            resolve();
        };
        document.head.appendChild(script);
    }

    loadOrbitControlsFromCDN(resolve) {
        const orbitScript = document.createElement('script');
        orbitScript.src = 'https://cdn.jsdelivr.net/npm/three@r128/examples/js/controls/OrbitControls.js';
        orbitScript.onload = () => {
            console.log('OrbitControls loaded from CDN');
            this.loadOBJLoaderFromCDN(resolve);
        };
        orbitScript.onerror = () => {
            console.warn('Failed to load OrbitControls from CDN');
            this.loadOBJLoaderFromCDN(resolve);
        };
        document.head.appendChild(orbitScript);
    }

    loadOBJLoaderFromCDN(resolve) {
        const objLoaderScript = document.createElement('script');
        objLoaderScript.src = 'https://cdn.jsdelivr.net/npm/three@r128/examples/js/loaders/OBJLoader.js';
        objLoaderScript.onload = () => {
            console.log('OBJLoader loaded from CDN');
            this.threeLoaded = true;
            resolve();
        };
        objLoaderScript.onerror = () => {
            console.warn('Failed to load OBJLoader from CDN');
            resolve();
        };
        document.head.appendChild(objLoaderScript);
    }
    ensureOptionalUI() {
        const meshLod = document.getElementById('meshLod');
        if (meshLod && !meshLod.dataset.bound) {
            meshLod.dataset.bound = 'true';
            meshLod.addEventListener('change', (e) => {
                this.currentMeshLod = e.target.value;
                this.reloadMeshForSelectedLod();
            });
        }
    }
    setupEventListeners() {
        const caseIdEl = document.getElementById('caseId');
        if (caseIdEl) {
            caseIdEl.addEventListener('keyup', () => this.updateSubmitButtonState());
        }
        document.querySelectorAll('.file-input').forEach(input => {
            input.addEventListener('change', (e) => this.handleFileSelect(e));
        });
        document.querySelectorAll('.drop-zone').forEach(zone => {
            zone.addEventListener('click', () => {
                const modality = zone.getAttribute('data-modality');
                const input = document.querySelector(`.file-input[data-modality="${modality}"]`);
                if (input) input.click();
            });
        });
        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submitAnalysis());
        }
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearAll());
        }
        document.querySelectorAll('.report-tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });
        const controlReset = document.getElementById('controlReset');
        if (controlReset) {
            controlReset.addEventListener('click', () => this.resetView());
        }
        const controlCenter = document.getElementById('controlCenter');
        if (controlCenter) {
            controlCenter.addEventListener('click', () => this.fitCameraToObject(this.mesh));
        }
        const controlDownload = document.getElementById('controlDownload');
        if (controlDownload) {
            controlDownload.addEventListener('click', () => this.downloadMesh());
        }
        const downloadReport = document.getElementById('downloadReport');
        if (downloadReport) {
            downloadReport.addEventListener('click', () => this.downloadFile('report'));
        }
        const downloadPrediction = document.getElementById('downloadPrediction');
        if (downloadPrediction) {
            downloadPrediction.addEventListener('click', () => this.downloadFile('pred_post'));
        }
        const downloadMesh = document.getElementById('downloadMesh');
        if (downloadMesh) {
            downloadMesh.addEventListener('click', () => this.downloadFile('mesh'));
        }
    }
    handleDragDrop() {
        document.querySelectorAll('.drop-zone').forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('drag-over');
            });
            zone.addEventListener('dragleave', () => {
                zone.classList.remove('drag-over');
            });
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');
                const modality = zone.getAttribute('data-modality');
                const input = document.querySelector(`.file-input[data-modality="${modality}"]`);
                if (input && e.dataTransfer.files.length > 0) {
                    input.files = e.dataTransfer.files;
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
    handleFileSelect(event) {
        const input = event.target;
        const modality = input.getAttribute('data-modality');
        const file = input.files[0];
        if (!file) return;
        if (!file.name.endsWith('.nii') && !file.name.endsWith('.nii.gz')) {
            this.showToast(`Invalid file format. Please upload .nii or .nii.gz files`, 'error');
            return;
        }
        const maxSize = 500 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showToast(`File size exceeds 500MB limit`, 'error');
            return;
        }
        this.uploadedFiles[modality] = {
            file: file,
            name: file.name,
            size: file.size
        };
        this.updateModality(modality);
        this.updateSubmitButtonState();
        this.showToast(`${modality.toUpperCase()} uploaded successfully`, 'success');
    }
    updateModality(modality) {
        const card = document.querySelector(`.modality-card[data-modality="${modality}"]`);
        if (!card) return;
        const dropZone = card.querySelector('.drop-zone');
        const fileInfo = card.querySelector('.file-info');
        const file = this.uploadedFiles[modality];
        if (!file) return;
        card.classList.add('active');
        if (dropZone) dropZone.style.display = 'none';
        if (fileInfo) {
            fileInfo.style.display = 'block';
            const nameEl = fileInfo.querySelector('.file-name');
            const sizeEl = fileInfo.querySelector('.file-size');
            if (nameEl) nameEl.textContent = file.name;
            if (sizeEl) sizeEl.textContent = `${this.formatBytes(file.size)}`;
        }
    }
    clearAll() {
        this.stopPolling();
        this.currentJobId = null;
        this.reportData = null;
        const resultsContainer = document.getElementById('resultsContainer');
        const emptyState = document.getElementById('emptyState');
        const statusCard = document.getElementById('statusCard');
        this.uploadedFiles = {};
        document.querySelectorAll('.modality-card').forEach(card => {
            card.classList.remove('active');
            const dropZone = card.querySelector('.drop-zone');
            const fileInfo = card.querySelector('.file-info');
            const status = card.querySelector('.modality-status');
            if (dropZone) dropZone.style.display = 'flex';
            if (fileInfo) fileInfo.style.display = 'none';
            if (status) status.textContent = '○';
        });
        const caseIdEl = document.getElementById('caseId');
        if (caseIdEl) caseIdEl.value = '';
        if (resultsContainer) resultsContainer.style.display = 'none';
        if (statusCard) statusCard.style.display = 'none';
        if (emptyState) emptyState.style.display = 'block';
        this.clearPreviewImages();
        this.destroy3DViewer();
        this.clearTabContents();
        this.updateSubmitButtonState();
        this.showToast('All uploads cleared', 'success');
    }

    clearTabContents() {
        const detailsContent = document.getElementById('detailsContent');
        if (detailsContent) detailsContent.innerHTML = '';

        const volumesChart = document.getElementById('volumesChart');
        if (volumesChart) volumesChart.innerHTML = '';

        const previewContent = document.getElementById('previewContent');
        if (previewContent) previewContent.innerHTML = '';
    }

    /* =====================================
       SUBMISSION & ANALYSIS
       ===================================== */

    updateSubmitButtonState() {
        const caseId = (document.getElementById('caseId')?.value || '').trim();
        const fileCount = Object.keys(this.uploadedFiles).length;
        const submitBtn = document.getElementById('submitBtn');

        if (!submitBtn) return;

        const isValid = !!caseId && fileCount >= 3;
        submitBtn.disabled = !isValid;
    }

    async submitAnalysis() {
        const caseId = (document.getElementById('caseId')?.value || '').trim();
        const fileCount = Object.keys(this.uploadedFiles).length;

        if (!caseId || fileCount < 3) {
            this.showToast('Please enter Case ID and upload at least 3 modalities', 'error');
            return;
        }
        this.stopPolling();
        this.currentJobId = null;
        this.reportData = null;
        this.clearPreviewImages();
        this.destroy3DViewer();
        const formData = new FormData();
        formData.append('case_id', caseId);

        for (const [modality, fileObj] of Object.entries(this.uploadedFiles)) {
            formData.append(modality, fileObj.file);
        }

        const enableSynthesis = document.getElementById('enableSynthesis');
        if (enableSynthesis) {
            formData.append('enable_synthesis', enableSynthesis.checked);
        }

        const generateMesh = document.getElementById('generateMesh');
        if (generateMesh) {
            formData.append('generate_mesh', generateMesh.checked);
        }

        const synSteps = document.getElementById('synSteps');
        if (synSteps && synSteps.value) {
            formData.append('syn_steps', parseInt(synSteps.value));
        }

        this.showStatusCard();
        this.statusStartTime = Date.now();

        try {
            const response = await fetch('/jobs', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                this.showToast(`Submission failed: ${err.detail || 'Unknown error'}`, 'error');
                return;
            }

            const data = await response.json();
            this.currentJobId = data.job_id;
            this.updateStatus('Job submitted. Processing...', 'queued');
            this.pollJobStatus();
        } catch (error) {
            this.showToast(`Network error: ${error.message}`, 'error');
            const statusCard = document.getElementById('statusCard');
            if (statusCard) statusCard.style.display = 'none';
        }
    }

    async pollJobStatus() {
        if (!this.currentJobId) return;

        const jobIdAtRequestTime = this.currentJobId;

        try {
            const response = await fetch(`/jobs/${this.currentJobId}/status`, {
                cache: 'no-store'
            });

            if (!response.ok) {
                this.showToast('Failed to fetch job status', 'error');
                return;
            }

            const job = await response.json();

            if (jobIdAtRequestTime !== this.currentJobId) {
                return;
            }

            this.updateStatus(
                job.status === 'completed' 
                    ? 'Analysis complete!' 
                    : `Processing: ${job.status}...`,
                job.status
            );

            if (job.status === 'failed') {
                this.showToast(`Job failed: ${job.error || 'Unknown error'}`, 'error');
                return;
            }

            if (job.status === 'completed') {
                await this.showResults();
                return;
            }

            this.pollTimer = setTimeout(() => this.pollJobStatus(), 2000);
        } catch (error) {
            this.showToast(`Polling error: ${error.message}`, 'error');
        }
    }

    /* =====================================
       STATUS & UI UPDATES
       ===================================== */

    showStatusCard() {
        const statusCard = document.getElementById('statusCard');
        const emptyState = document.getElementById('emptyState');

        if (statusCard) statusCard.style.display = 'block';
        if (emptyState) emptyState.style.display = 'none';
    }

    updateStatus(message, status) {
        const started = this.statusStartTime || Date.now();
        const elapsedSeconds = Math.floor((Date.now() - started) / 1000);
        const minutes = Math.floor(elapsedSeconds / 60);
        const seconds = elapsedSeconds % 60;
        const timeStr = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

        const titleEl = document.getElementById('statusTitle');
        const messageEl = document.getElementById('statusMessage');
        const timeEl = document.getElementById('statusTime');
        const progressEl = document.getElementById('progressFill');

        if (titleEl) titleEl.textContent = `Status: ${status.charAt(0).toUpperCase() + status.slice(1)}`;
        if (messageEl) messageEl.textContent = message;
        if (timeEl) timeEl.textContent = timeStr;

        const progressMap = {
            queued: 10,
            running: 50,
            completed: 100,
            failed: 100
        };

        const width = progressMap[status] || 10;
        if (progressEl) progressEl.style.width = `${width}%`;
    }

    async showResults() {
        const statusCard = document.getElementById('statusCard');
        const resultsContainer = document.getElementById('resultsContainer');
        const emptyState = document.getElementById('emptyState');

        await this.loadResults();
        await this.initalize3DViewer();

        if (statusCard) statusCard.style.display = 'none';
        if (emptyState) emptyState.style.display = 'none';
        if (resultsContainer) resultsContainer.style.display = 'grid';
    }

    async loadResults() {
        const response = await fetch(`/jobs/${this.currentJobId}/report`, {
            cache: 'no-store',
            headers: {
                Accept: 'application/json'
            }
        });

        if (!response.ok) {
            this.showToast('Failed to load report', 'error');
            return;
        }

        const report = await response.json();
        this.reportData = report;

        const metricStatus = document.getElementById('metricStatus');
        if (metricStatus) {
            metricStatus.textContent = report.status ? report.status.toUpperCase() : 'UNKNOWN';
        }

        const timeSeconds = Math.floor((Date.now() - (this.statusStartTime || Date.now())) / 1000);
        const minutes = Math.floor(timeSeconds / 60);
        const seconds = timeSeconds % 60;

        const metricTime = document.getElementById('metricTime');
        if (metricTime) {
            metricTime.textContent = `${minutes}m ${seconds}s`;
        }

        const metricModel = document.getElementById('metricModel');
        if (metricModel) {
            metricModel.textContent = report.metadata?.uses_monai ? 'UNet3D (MONAI)' : 'UNet3D';
        }

        const metricSynthesis = document.getElementById('metricSynthesis');
        if (metricSynthesis) {
            metricSynthesis.textContent = report.synthesis_status || 'Off';
        }

        this.updateDetailsTab(report);
        this.updateVolumesTab(report);
        this.updatePreviewTab(report);

        return report;
    }

    updateDetailsTab(report) {
        const detailsContent = document.getElementById('detailsContent');
        if (!detailsContent) return;

        let html = '<div class="details-list">';

        html += `<div class="detail-item">
            <span class="detail-label">Case ID:</span>
            <span class="detail-value">${this.escapeHtml(report.case_id || '-')}</span>
        </div>`;

        html += `<div class="detail-item">
            <span class="detail-label">Status:</span>
            <span class="detail-value">${this.escapeHtml(report.status || '-')}</span>
        </div>`;

        if (report.missing_flags) {
            const flags = Object.entries(report.missing_flags)
                .filter(([_, v]) => v)
                .map(([k]) => k)
                .join(', ');
            if (flags) {
                html += `<div class="detail-item">
                    <span class="detail-label">Missing Modalities:</span>
                    <span class="detail-value">${this.escapeHtml(flags)}</span>
                </div>`;
            }
        }

        if (report.synthesis_status) {
            html += `<div class="detail-item">
                <span class="detail-label">Synthesis:</span>
                <span class="detail-value">${this.escapeHtml(report.synthesis_status)}</span>
            </div>`;
        }

        if (report.downsample_factor !== undefined && report.downsample_factor !== null) {
            html += `<div class="detail-item">
                <span class="detail-label">Downsample Factor:</span>
                <span class="detail-value">${report.downsample_factor}</span>
            </div>`;
        }

        if (report.spacing_mm) {
            html += `<div class="detail-item">
                <span class="detail-label">Spacing (mm):</span>
                <span class="detail-value">${report.spacing_mm}</span>
            </div>`;
        }

        html += '</div>';
        detailsContent.innerHTML = html;
    }

    updateVolumesTab(report) {
        const volumesChart = document.getElementById('volumesChart');
        if (!volumesChart) return;

        if (!report.region_volumes_mm3) {
            volumesChart.innerHTML = '<p>No volume data available</p>';
            return;
        }

        let html = '<div class="volumes-list">';
        for (const [region, volume] of Object.entries(report.region_volumes_mm3)) {
            const displayName = region === 'ET' ? 'Enhancing Tumor' : 
                              region === 'TC' ? 'Tumor Core' : 
                              region === 'WT' ? 'Whole Tumor' : region;
            html += `<div class="volume-item">
                <span class="volume-label">${this.escapeHtml(displayName)}:</span>
                <span class="volume-value">${(volume || 0).toFixed(2)} mm³</span>
            </div>`;
        }
        html += '</div>';

        volumesChart.innerHTML = html;
    }

    updatePreviewTab(report) {
        const preview = report.preview || {};
        const previewContent = document.getElementById('previewContent');

        const axialImg = document.getElementById('previewAxial');
        const coronalImg = document.getElementById('previewCoronal');
        const sagittalImg = document.getElementById('previewSagittal');

        if (axialImg) {
            axialImg.src = preview.axial || '';
            axialImg.style.display = preview.axial ? 'block' : 'none';
        }

        if (coronalImg) {
            coronalImg.src = preview.coronal || '';
            coronalImg.style.display = preview.coronal ? 'block' : 'none';
        }

        if (sagittalImg) {
            sagittalImg.src = preview.sagittal || '';
            sagittalImg.style.display = preview.sagittal ? 'block' : 'none';
        }

        if (previewContent && !axialImg && !coronalImg && !sagittalImg) {
            previewContent.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 2rem;">
                    <p style="color: #888;">Preview images not available</p>
                </div>
            `;
        }
    }

    clearPreviewImages() {
        ['previewAxial', 'previewCoronal', 'previewSagittal'].forEach(id => {
            const img = document.getElementById(id);
            if (img) {
                img.src = '';
                img.style.display = 'none';
            }
        });

        const previewContent = document.getElementById('previewContent');
        if (previewContent) previewContent.innerHTML = '';
    }

    switchTab(tabName) {
        document.querySelectorAll('.report-tab').forEach(tab => {
            tab.classList.remove('active');
        });

        const targetTab = document.querySelector(`.report-tab[data-tab="${tabName}"]`);
        if (targetTab) {
            targetTab.classList.add('active');
        }

        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        const activeContent = document.getElementById(tabName + 'Tab');
        if (activeContent) {
            activeContent.classList.add('active');
        }
    }

    /* =====================================
       3D VISUALIZATION
       ===================================== */

    async initalize3DViewer() {
        const container = document.getElementById('threejsContainer');
        if (!container) {
            console.warn('Container not found for 3D viewer');
            return;
        }
        this.destroy3DViewer();
        container.innerHTML = '';
        // Wait for THREE.js to be loaded
        if (!window.THREE || !this.threeLoaded) {
            await this.ensureThreeJSLoaded();
        }
        if (!window.THREE) {
            this.createPlaceholderGeometry('THREE.js library not available');
            return;
        }
        try {
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x0f172a);
            const width = Math.max(container.clientWidth || 640, 320);
            const height = Math.max(container.clientHeight || 480, 240);
            this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 10000);
            this.camera.position.set(0, 0, 200);
            this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(window.devicePixelRatio || 1);
            this.renderer.sortObjects = true;
            container.appendChild(this.renderer.domElement);
            if (window.THREE.OrbitControls) {
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.08;
                this.controls.enableZoom = true;
                this.controls.autoRotate = true;
                this.controls.autoRotateSpeed = 0.8;
            } else {
                console.warn('OrbitControls not available');
            }

            // Ambient light for base illumination
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            this.scene.add(ambientLight);

            // Hemisphere light for natural sky/ground lighting
            const hemiLight = new THREE.HemisphereLight(0x88ccff, 0x444422, 0.4);
            this.scene.add(hemiLight);

            // Key light
            const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.7);
            directionalLight1.position.set(150, 200, 150);
            this.scene.add(directionalLight1);

            // Fill light
            const directionalLight2 = new THREE.DirectionalLight(0x8899cc, 0.4);
            directionalLight2.position.set(-100, -50, 100);
            this.scene.add(directionalLight2);

            // Rim/back light for edge definition
            const directionalLight3 = new THREE.DirectionalLight(0x6688cc, 0.3);
            directionalLight3.position.set(0, -100, -150);
            this.scene.add(directionalLight3);

            await this.reloadMeshForSelectedLod();

            // Wire up brain toggle
            const brainToggle = document.getElementById('brainToggle');
            if (brainToggle && !brainToggle.dataset.bound) {
                brainToggle.dataset.bound = 'true';
                brainToggle.addEventListener('change', (e) => {
                    if (this.brainMeshObject) {
                        this.brainMeshObject.visible = e.target.checked;
                    }
                });
            }

            const animate = () => {
                this.animationFrameId = requestAnimationFrame(animate);
                if (this.controls) this.controls.update();
                this.renderer.render(this.scene, this.camera);
            };
            animate();
            this.resizeHandler = () => this.onWindowResize();
            window.addEventListener('resize', this.resizeHandler);
        } catch (error) {
            console.error('Error initializing 3D viewer:', error);
            this.createPlaceholderGeometry(`Initialization error: ${error.message}`);
        }
    }
    async reloadMeshForSelectedLod() {
        if (!this.scene) return;

        // Remove existing brain mesh
        if (this.brainMeshObject) {
            this.scene.remove(this.brainMeshObject);
            this.disposeObject(this.brainMeshObject);
            this.brainMeshObject = null;
        }
        // Remove existing grid box
        if (this.gridBox) {
            this.scene.remove(this.gridBox);
            this.gridBox = null;
        }
        // Remove existing region group
        const objectToRemove = this.regionGroup || this.mesh;
        if (objectToRemove) {
            this.scene.remove(objectToRemove);
            this.disposeObject(objectToRemove);
        }
        this.mesh = null;
        this.regionGroup = null;

        const regionUrls = this.getRegionMeshUrlsForLod(this.currentMeshLod);
        if (!regionUrls) {
            this.createPlaceholderGeometry('No region meshes available');
            return;
        }
        await this.loadRegionMeshes(regionUrls);
    }

    getBrainMeshUrlForLod(lod) {
        if (!this.reportData) return null;
        const requested = String(lod || 'low').toLowerCase();
        const viewer = this.reportData.viewer;
        if (!viewer || !viewer.brain) return null;
        return viewer.brain[requested] || null;
    }

    getRegionMeshUrlsForLod(lod) {
        if (!this.reportData) return null;
        const requested = String(lod || 'low').toLowerCase();
        const regions = this.reportData.viewer?.regions;
        if (!regions) return null;
        return {
            wt: regions.wt?.[requested] || null,
            tc: regions.tc?.[requested] || null,
            et: regions.et?.[requested] || null
        };
    }

    createGridBox(size) {
        // Create a wireframe grid cube as spatial reference (like the reference image)
        const gridGroup = new THREE.Group();

        const gridSize = size * 1.3;
        const divisions = 8;
        const gridColor = new THREE.Color(0x334455);

        // Bottom grid
        const bottomGrid = new THREE.GridHelper(gridSize, divisions, gridColor, gridColor);
        bottomGrid.position.y = -gridSize / 2;
        gridGroup.add(bottomGrid);

        // Back grid (rotated)
        const backGrid = new THREE.GridHelper(gridSize, divisions, gridColor, gridColor);
        backGrid.rotation.x = Math.PI / 2;
        backGrid.position.z = -gridSize / 2;
        gridGroup.add(backGrid);

        // Left grid (rotated)
        const leftGrid = new THREE.GridHelper(gridSize, divisions, gridColor, gridColor);
        leftGrid.rotation.z = Math.PI / 2;
        leftGrid.position.x = -gridSize / 2;
        gridGroup.add(leftGrid);

        // Wireframe bounding box
        const boxGeom = new THREE.BoxGeometry(gridSize, gridSize, gridSize);
        const boxEdges = new THREE.EdgesGeometry(boxGeom);
        const boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({
            color: 0x445566,
            transparent: true,
            opacity: 0.4
        }));
        gridGroup.add(boxLine);

        return gridGroup;
    }

    async loadRegionMeshes(regionUrls) {
        try {
            if (!window.THREE || !window.THREE.OBJLoader) {
                this.createPlaceholderGeometry('OBJLoader not available');
                return;
            }

            const group = new THREE.Group();
            const loader = new THREE.OBJLoader();

            // ---- Load brain mesh first (transparent outer shell) ----
            const brainUrl = this.getBrainMeshUrlForLod(this.currentMeshLod);
            if (brainUrl) {
                try {
                    const brainResponse = await fetch(brainUrl);
                    if (brainResponse.ok) {
                        const brainText = await brainResponse.text();
                        const brainObject = loader.parse(brainText);

                        brainObject.traverse((child) => {
                            if (child.isMesh) {
                                child.material = new THREE.MeshPhongMaterial({
                                    color: 0x44eeff,
                                    emissive: 0x0a2233,
                                    transparent: true,
                                    opacity: 0.12,
                                    depthWrite: false,
                                    shininess: 30,
                                    side: THREE.DoubleSide,
                                    blending: THREE.NormalBlending,
                                });
                                child.renderOrder = 0;
                            }
                        });

                        this.brainMeshObject = brainObject;
                        this.scene.add(brainObject);

                        // Sync with toggle state
                        const brainToggle = document.getElementById('brainToggle');
                        if (brainToggle) {
                            brainObject.visible = brainToggle.checked;
                        }
                        console.log('Brain mesh loaded successfully');
                    }
                } catch (brainErr) {
                    console.warn('Failed to load brain mesh:', brainErr);
                }
            }

            // ---- Load tumor region meshes ----
            // Ordered from outermost (WT) to innermost (ET) with increasing opacity
            const configs = [
                { key: 'wt', color: 0x00e5ff, emissive: 0x003344, opacity: 0.22, shininess: 40, renderOrder: 1, name: 'Whole Tumor' },
                { key: 'tc', color: 0xcc66ff, emissive: 0x220044, opacity: 0.50, shininess: 60, renderOrder: 2, name: 'Tumor Core' },
                { key: 'et', color: 0xff4466, emissive: 0x330011, opacity: 0.85, shininess: 90, renderOrder: 3, name: 'Enhancing Tumor' }
            ];

            let loadedAny = false;

            for (const cfg of configs) {
                const url = regionUrls[cfg.key];
                if (!url) continue;

                const response = await fetch(url);
                if (!response.ok) continue;

                const text = await response.text();
                const object = loader.parse(text);

                object.traverse((child) => {
                    if (child.isMesh) {
                        child.material = new THREE.MeshPhongMaterial({
                            color: cfg.color,
                            emissive: cfg.emissive,
                            transparent: true,
                            opacity: cfg.opacity,
                            depthWrite: cfg.key === 'et',  // Only innermost writes depth
                            shininess: cfg.shininess,
                            side: THREE.DoubleSide,
                            blending: THREE.NormalBlending,
                        });
                        child.renderOrder = cfg.renderOrder;
                    }
                });

                group.add(object);
                loadedAny = true;
                console.log(`Loaded ${cfg.name} mesh`);
            }

            if (!loadedAny) {
                this.createPlaceholderGeometry('No region mesh could be loaded');
                return;
            }

            this.regionGroup = group;
            this.mesh = group; // để các nút reset/center vẫn dùng lại logic cũ
            this.scene.add(group);

            // Add spatial reference grid box based on the bounding box of all content
            const allContent = new THREE.Group();
            if (this.brainMeshObject) allContent.add(this.brainMeshObject.clone());
            allContent.add(group.clone());
            const box = new THREE.Box3().setFromObject(allContent);
            if (!box.isEmpty()) {
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const gridBox = this.createGridBox(maxDim);
                const center = box.getCenter(new THREE.Vector3());
                gridBox.position.copy(center);
                this.gridBox = gridBox;
                this.scene.add(gridBox);
            }

            // Fit camera to include brain + tumor
            const fitTarget = this.brainMeshObject || group;
            this.fitCameraToObject(fitTarget);
        } catch (error) {
            console.error('Error loading region meshes:', error);
            this.createPlaceholderGeometry(`Region mesh load error: ${error.message}`);
        }
    }
    getMeshUrlForLod(lod) {
        if (!this.reportData) return null;
        const requested = String(lod || 'low').toLowerCase();
        const downloads = this.reportData.downloads || {};
        if (requested === 'high' && downloads.brain_mesh_high) return downloads.brain_mesh_high;
        if (requested === 'medium' && downloads.brain_mesh_medium) return downloads.brain_mesh_medium;
        if (downloads.brain_mesh_low) return downloads.brain_mesh_low;
        return this.currentJobId
            ? `/jobs/${this.currentJobId}/file/brain_mesh?lod=${encodeURIComponent(requested)}`
            : null;
    }

    async loadMesh(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                this.createPlaceholderGeometry(`Mesh fetch failed: ${response.status}`);
                return;
            }

            const text = await response.text();

            if (window.THREE && window.THREE.OBJLoader) {
                const loader = new THREE.OBJLoader();
                const object = loader.parse(text);

                object.traverse((child) => {
                    if (child.isMesh) {
                        child.material = new THREE.MeshPhongMaterial({
                            color: 0x3b82f6,
                            emissive: 0x1e40af,
                            shininess: 100
                        });
                    }
                });

                this.mesh = object;
                this.scene.add(object);
                this.fitCameraToObject(object);
                return;
            }
            this.createPlaceholderGeometry('OBJLoader not available');
        } catch (error) {
            console.error('Error loading mesh:', error);
            this.createPlaceholderGeometry(`Load error: ${error.message}`);
        }
    }
    parseSTLBinary(arrayBuffer) {
        const view = new DataView(arrayBuffer);
        const triangles = view.getUint32(80, true);
        const geometry = new THREE.BufferGeometry();
        const vertices = [];
        const normals = [];
        let offset = 84;
        for (let i = 0; i < triangles; i++) {
            const nx = view.getFloat32(offset, true);
            const ny = view.getFloat32(offset + 4, true);
            const nz = view.getFloat32(offset + 8, true);
            offset += 12;
            for (let j = 0; j < 3; j++) {
                vertices.push(
                    view.getFloat32(offset, true),
                    view.getFloat32(offset + 4, true),
                    view.getFloat32(offset + 8, true)
                );
                normals.push(nx, ny, nz);
                offset += 12;
            }

            offset += 2; // attribute byte count
        }
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
        geometry.computeBoundingBox();
        return geometry;
    }
    createPlaceholderGeometry(reason = '') {
        if (!window.THREE || !this.scene) return;
        const group = new THREE.Group();
        const etGeom = new THREE.SphereGeometry(24, 32, 32);
        const etMat = new THREE.MeshPhongMaterial({ color: 0xef4444 });
        const etMesh = new THREE.Mesh(etGeom, etMat);
        etMesh.position.set(0, 0, -10);
        group.add(etMesh);
        const tcGeom = new THREE.CylinderGeometry(34, 34, 70, 32);
        const tcMat = new THREE.MeshPhongMaterial({ color: 0xa78bfa, transparent: true, opacity: 0.75 });
        const tcMesh = new THREE.Mesh(tcGeom, tcMat);
        tcMesh.rotation.x = Math.PI / 2;
        group.add(tcMesh);
        const wtGeom = new THREE.BoxGeometry(95, 110, 80);
        const wtMat = new THREE.MeshPhongMaterial({
            color: 0x34d399,
            emissive: 0x052e2b,
            wireframe: true,
            transparent: true,
            opacity: 0.35
        });
        const wtMesh = new THREE.Mesh(wtGeom, wtMat);
        group.add(wtMesh);
        this.mesh = group;
        this.scene.add(group);
        this.fitCameraToObject(group);
        const container = document.getElementById('threejsContainer');
        const noteId = 'viewerFallbackNote';
        let note = document.getElementById(noteId);
        if (!note && container) {
            note = document.createElement('div');
            note.id = noteId;
            note.style.cssText = 'position:absolute; top:10px; right:10px; background:#f97316; color:white; padding:8px 12px; border-radius:4px; font-size:12px; z-index:100;';
            note.textContent = `⚠ Placeholder (${reason})`;
            container.appendChild(note);
        }
        if (note) {
            note.style.display = 'block';
        }
    }
    fitCameraToObject(object) {
        if (!this.camera || !object) return;
        const box = new THREE.Box3().setFromObject(object);
        if (box.isEmpty()) return;
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const fov = this.camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraZ *= 1.8;
        this.camera.position.set(center.x, center.y, center.z + cameraZ);
        this.camera.near = Math.max(0.1, maxDim / 100);
        this.camera.far = Math.max(1000, maxDim * 20);
        this.camera.updateProjectionMatrix();
        if (this.controls) {
            this.controls.target.copy(center);
            this.controls.update();
        }
    }
    resetView() {
        if (this.mesh) {
            this.fitCameraToObject(this.mesh);
        }
    }
    downloadMesh() {
        if (!this.reportData) {
            this.showToast('Mesh not available for download', 'error');
            return;
        }
        const lod = String(this.currentMeshLod || 'low').toLowerCase();
        const url = this.reportData.downloads?.[`wt_mesh_${lod}`];
        if (url) {
            window.open(url, '_blank');
        } else {
            this.showToast('WT mesh not available for download', 'error');
        }
    }

    onWindowResize() {
        if (!this.renderer || !this.camera) return;
        const container = document.getElementById('threejsContainer');
        if (!container) return;

        const width = Math.max(container.clientWidth || 640, 320);
        const height = Math.max(container.clientHeight || 480, 240);

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    destroy3DViewer() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.resizeHandler) {
            window.removeEventListener('resize', this.resizeHandler);
            this.resizeHandler = null;
        }
        if (this.controls) {
            this.controls.dispose();
            this.controls = null;
        }
        if (this.brainMeshObject) {
            this.disposeObject(this.brainMeshObject);
            this.brainMeshObject = null;
        }
        if (this.gridBox) {
            this.gridBox = null;
        }
        const objectToDispose = this.regionGroup || this.mesh;
        if (objectToDispose) {
            this.disposeObject(objectToDispose);
        }
        this.mesh = null;
        this.regionGroup = null;
        this.camera = null;
    }

    disposeObject(object) {
        object.traverse((node) => {
            if (node instanceof THREE.Mesh) {
                if (node.geometry) node.geometry.dispose();
                if (node.material) {
                    if (Array.isArray(node.material)) {
                        node.material.forEach(m => m.dispose());
                    } else {
                        node.material.dispose();
                    }
                }
            }
        });
    }

    /* =====================================
       FILE DOWNLOADS
       ===================================== */

    downloadFile(kind) {
        if (!this.currentJobId) {
            this.showToast('No job available for download', 'error');
            return;
        }

        const url = `/jobs/${this.currentJobId}/file/${kind}`;
        const link = document.createElement('a');
        link.href = url;
        link.download = `${kind}_${this.currentJobId}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    /* =====================================
       UTILITIES
       ===================================== */

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            padding: 12px 16px;
            margin: 8px;
            border-radius: 4px;
            background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#3b82f6'};
            color: white;
            animation: slideIn 0.3s ease-out;
        `;

        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s';
            setTimeout(() => container.removeChild(toast), 300);
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

    escapeHtml(value) {
        if (!value) return '';
        const div = document.createElement('div');
        div.textContent = String(value);
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new BrainAnalysisApp();
});