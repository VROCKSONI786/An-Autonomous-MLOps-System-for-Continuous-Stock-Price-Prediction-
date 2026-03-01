pipeline {
    agent any

    // ── Triggers ──────────────────────────────────────────────────────────────
    triggers {
        // Primary: 6 PM IST (12:30 UTC) every weekday after market close
        cron('30 12 * * 1-5')
    }

    environment {
        PROJECT_ROOT   = "${WORKSPACE}"
        VENV           = "${WORKSPACE}/venv"
        PY             = "${WORKSPACE}/venv/bin/python"
        PIP            = "${WORKSPACE}/venv/bin/pip"
        MLFLOW_URI     = "http://localhost:5000"
        FRED_API_KEY   = credentials('FRED_API_KEY')     // register in Jenkins → Manage → Credentials
        GEMINI_API_KEY = credentials('GEMINI_API_KEY')
        // Write .env so Python dotenv can pick them up
        DOTENV_FILE    = "${WORKSPACE}/.env"
    }

    options {
        timeout(time: 4, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '30'))
        timestamps()
        skipDefaultCheckout(false)
        // Prevent concurrent builds — retraining must run sequentially
        disableConcurrentBuilds()
    }

    parameters {
        booleanParam(
            name:         'FORCE_RETRAIN',
            defaultValue: false,
            description:  'Force retrain all symbols regardless of drift/performance'
        )
        string(
            name:         'SYMBOL',
            defaultValue: '',
            description:  'Retrain a specific symbol only (e.g. TCS.NS). Leave blank for all.'
        )
        booleanParam(
            name:         'SKIP_DATA_COLLECTION',
            defaultValue: false,
            description:  'Skip data collection (use existing raw data)'
        )
    }

    stages {

        // ──────────────────────────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
                echo "Branch: ${env.GIT_BRANCH} | Commit: ${env.GIT_COMMIT?.take(8)}"
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        stage('Setup Environment') {
            steps {
                sh '''
                    # Write secrets to .env so Python dotenv picks them up
                    echo "GEMINI_API_KEY=${GEMINI_API_KEY}" > ${DOTENV_FILE}
                    echo "FRED_API_KEY=${FRED_API_KEY}"    >> ${DOTENV_FILE}

                    # Create venv if not exists
                    [ -d "${VENV}" ] || python3 -m venv ${VENV}

                    ${PIP} install --upgrade pip --quiet
                    ${PIP} install -r requirements.txt --quiet

                    echo "=== Python environment ready ==="
                    ${PY} --version
                    ${PY} -c "import tensorflow, mlflow, streamlit; print('Core packages OK')"
                '''
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        stage('Collect Market & Economic Data') {
            when {
                expression { return !params.SKIP_DATA_COLLECTION }
            }
            steps {
                sh '''
                    cd ${PROJECT_ROOT}
                    echo "--- Collecting stock prices, fundamentals, news ---"
                    ${PY} -m src.data_collection.data_orchestrator

                    echo "--- Collecting global economic indicators (FRED) ---"
                    ${PY} -m src.data_collection.economic_data_collector
                '''
            }
            post {
                failure {
                    echo "Data collection failed. Continuing with existing data if available."
                }
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        stage('Preprocess Data') {
            steps {
                sh '''
                    cd ${PROJECT_ROOT}
                    ${PY} -m src.preprocessing.data_preprocessor
                    echo "=== Preprocessed data ready ==="
                    ls -lh data/processed/
                '''
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        stage('Drift Detection') {
            steps {
                sh '''
                    cd ${PROJECT_ROOT}
                    ${PY} -m src.monitoring.drift_detector
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'data/reports/drift_summary_*.json',
                                     allowEmptyArchive: true
                    archiveArtifacts artifacts: 'data/reports/*_drift_*.html',
                                     allowEmptyArchive: true
                }
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        stage('Performance Monitoring') {
            steps {
                sh '''
                    cd ${PROJECT_ROOT}
                    ${PY} -m src.monitoring.performance_monitor
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'logs/performance_metrics_*.json',
                                     allowEmptyArchive: true
                }
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        stage('Retraining Decision & Execution') {
            steps {
                script {
                    def cmd = "${PY} ${PROJECT_ROOT}/src/retraining_scheduler.py"

                    if (params.FORCE_RETRAIN) {
                        cmd += " --force"
                        echo "FORCE_RETRAIN=true — retraining all symbols"
                    }
                    if (params.SYMBOL?.trim()) {
                        cmd += " --symbol ${params.SYMBOL}"
                        echo "Targeting symbol: ${params.SYMBOL}"
                    }

                    sh "cd ${PROJECT_ROOT} && ${cmd}"
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'logs/retrain_run_*.json',
                                     allowEmptyArchive: true
                    archiveArtifacts artifacts: 'models/saved_models/*.keras',
                                     allowEmptyArchive: true,
                                     fingerprint: true
                    archiveArtifacts artifacts: 'logs/plots/*.png',
                                     allowEmptyArchive: true
                }
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        stage('Post-Retrain Performance Check') {
            steps {
                sh '''
                    cd ${PROJECT_ROOT}
                    echo "=== Running performance check after retraining ==="
                    ${PY} -m src.monitoring.performance_monitor

                    echo "=== Latest performance snapshot ==="
                    ${PY} -c "
import json, glob
files = sorted(glob.glob('logs/performance_metrics_*.json'))
if files:
    with open(files[-1]) as f:
        metrics = json.load(f)
    print(f'Models evaluated: {len(metrics)}')
    for m in metrics:
        print(f\"  {m.get('symbol','?'):20s} | \
MAE=INR{m.get('mae_inr',0):.2f} | \
MAPE={m.get('mape_pct',0):.2f}% | \
R2={m.get('r2_score',0):.4f} | \
DirAcc={m.get('direction_accuracy_pct',0):.1f}% | \
ThreshAcc={m.get('threshold_accuracy_pct',0):.1f}%\")
else:
    print('No metrics file found')
"
                '''
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        stage('Deploy / Restart Streamlit') {
            when {
                anyOf { branch 'main'; branch 'master' }
            }
            steps {
                sh '''
                    echo "=== Restarting Streamlit app ==="
                    pkill -f "streamlit run" || true
                    sleep 3

                    nohup ${VENV}/bin/streamlit run \
                        ${PROJECT_ROOT}/streamlit_app/main.py \
                        --server.port 8501 \
                        --server.headless true \
                        --server.address 0.0.0.0 \
                        > ${PROJECT_ROOT}/logs/streamlit.log 2>&1 &

                    sleep 8
                    curl -sf http://localhost:8501/_stcore/health \
                        && echo "Streamlit is healthy." \
                        || echo "WARNING: Streamlit health check failed — check logs/streamlit.log"
                '''
            }
        }

    } // end stages

    // ── Post-pipeline ──────────────────────────────────────────────────────────
    post {
        success {
            echo """
╔══════════════════════════════════════╗
║  ✅  Pipeline completed successfully ║
╚══════════════════════════════════════╝
Build: #${env.BUILD_NUMBER}
URL:   ${env.BUILD_URL}
"""
        }
        failure {
            echo """
╔══════════════════════════════════════╗
║  ❌  Pipeline FAILED                 ║
╚══════════════════════════════════════╝
Build: #${env.BUILD_NUMBER}
Logs:  ${env.BUILD_URL}console
"""
        }
        always {
            // Clean up .env file so secrets aren't left on disk
            sh 'rm -f ${DOTENV_FILE} || true'

            archiveArtifacts artifacts: 'logs/streamlit.log',
                             allowEmptyArchive: true
        }
    }
}