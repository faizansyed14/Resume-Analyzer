# app/services/skill_library.py

class SkillLibrary:
    """
    Comprehensive skill library with categorized technical and soft skills.
    Used for accurate skill extraction and matching in resume and job description parsing.
    """
    
    def __init__(self):
        self.skill_categories = {
                  "programming_languages": [
          "Python", "Java", "JavaScript", "C++", "C#", "C", "PHP", "Ruby", "Go", "Rust",
          "Swift", "Kotlin", "Scala", "R", "MATLAB", "Perl", "Bash", "Shell Scripting", "TypeScript",
          "Objective-C", "Dart", "Lua", "Haskell", "Clojure", "Erlang", "Elixir", "F#",
          "Visual Basic", "VB.NET", "Assembly", "COBOL", "Fortran", "Pascal", "Delphi",
          "Groovy", "Julia", "Racket", "Scheme", "Prolog", "Smalltalk", "Ada", "ABAP",
          "PowerShell", "VBScript", "Tcl", "CoffeeScript", "Crystal", "Nim", "Zig",
          "OCaml", "ReasonML", "PureScript", "Idris", "Verilog", "VHDL", "SystemVerilog",
          "ActionScript", "AppleScript", "AutoHotkey", "AWK", "Boo", "Chapel", "D",
          "Eiffel", "Factor", "Forth", "Icon", "J", "Lasso", "Limbo", "Logo", "Max/MSP",
          "Miranda", "Modula-2", "Oberon", "Pike", "PostScript", "Q", "Red", "Ring",
          "S-Lang", "Seed7", "SPARK", "Standard ML", "Stata", "Turing", "Vala", "XQuery",
          "XSLT", "Yorick", "ZPL", "4th Dimension", "ALGOL", "APL", "BASIC", "BCPL",
          "Brainfuck", "C Shell", "CLIPS", "Common Lisp", "Curry", "Dylan", "Euphoria"
      ],

      "web_technologies": [
          "HTML", "HTML5", "CSS", "CSS3", "JavaScript", "React", "React.js", "Angular", 
          "Angular.js", "AngularJS", "Vue.js", "Vue", "Node.js", "Express", "Express.js",
          "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "Laravel", "Symfony", 
          "Ruby on Rails", "Rails", "ASP.NET", "ASP.NET Core", "jQuery", "Bootstrap",
          "Sass", "SCSS", "Less", "Stylus", "Webpack", "Gulp", "Grunt", "Babel",
          "Next.js", "Nuxt.js", "Gatsby", "Svelte", "SvelteKit", "Backbone.js", "Ember.js",
          "Meteor", "Koa", "Hapi", "Fastify", "Deno", "Bun", "Vite", "Parcel",
          "Rollup", "ESLint", "Prettier", "TypeScript", "CoffeeScript", "Elm",
          "PurgeCSS", "Tailwind CSS", "Bulma", "Foundation", "Materialize", "Semantic UI",
          "Chakra UI", "Ant Design", "Material-UI", "React Bootstrap", "Styled Components",
          "Emotion", "PostCSS", "Autoprefixer", "CORS", "JWT", "OAuth", "REST API",
          "GraphQL", "Apollo", "Relay", "Prisma", "Hasura", "Strapi", "Contentful",
          "Sanity", "Ghost", "WordPress", "Drupal", "Joomla", "Magento", "Shopify"
      ],

      "databases": [
          "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Oracle Database",
          "SQL Server", "Microsoft SQL Server", "Cassandra", "Redis", "Elasticsearch",
          "Neo4j", "DynamoDB", "Firebase", "Firestore", "CouchDB", "InfluxDB",
          "MariaDB", "DB2", "Sybase", "Access", "Microsoft Access", "HBase", "CouchBase",
          "Amazon RDS", "Amazon Aurora", "Google Cloud SQL", "Azure SQL Database",
          "BigQuery", "Snowflake", "Redshift", "Azure Synapse", "Apache Spark",
          "Apache Hive", "Apache Impala", "Presto", "Trino", "ClickHouse",
          "TimescaleDB", "ScyllaDB", "ArangoDB", "OrientDB", "Amazon Neptune",
          "Azure Cosmos DB", "Google Spanner", "FaunaDB", "SurrealDB", "EdgeDB",
          "RethinkDB", "RavenDB", "EventStore", "Apache Drill", "Apache Phoenix",
          "VoltDB", "MemSQL", "SingleStore", "TiDB", "CockroachDB", "YugabyteDB"
      ],

      "cloud_technologies": [
          "AWS", "Amazon Web Services", "Azure", "Microsoft Azure", "Google Cloud", 
          "Google Cloud Platform", "GCP", "Docker", "Kubernetes", "Terraform",
          "Ansible", "Jenkins", "Travis CI", "CircleCI", "GitLab CI", "GitHub Actions",
          "Heroku", "DigitalOcean", "Linode", "Cloudflare", "Lambda", "AWS Lambda",
          "EC2", "S3", "RDS", "EKS", "ECS", "CloudFormation", "CloudWatch",
          "Azure Functions", "Azure DevOps", "Google Functions", "Cloud Run",
          "Cloud Functions", "Kubernetes Engine", "App Engine", "Compute Engine",
          "Azure Container Instances", "Azure Kubernetes Service", "AKS",
          "OpenStack", "VMware", "Vagrant", "Packer", "Consul", "Vault",
          "Nomad", "Prometheus", "Grafana", "Jaeger", "Zipkin", "New Relic",
          "Datadog", "Splunk", "ELK Stack", "Elasticsearch", "Logstash", "Kibana",
          "Fluentd", "Istio", "Linkerd", "Envoy", "NGINX", "Apache", "HAProxy",
          "Load Balancer", "CloudFront", "CDN", "Route 53", "VPC", "IAM"
      ],

      "ai_ml": [
          "Machine Learning", "ML", "Deep Learning", "Artificial Intelligence", "AI",
          "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Sklearn", "Pandas", 
          "NumPy", "Matplotlib", "Seaborn", "Plotly", "OpenCV", "NLTK", "spaCy",
          "Transformers", "Hugging Face", "Neural Networks", "CNN", "Convolutional Neural Networks",
          "RNN", "Recurrent Neural Networks", "LSTM", "Long Short-Term Memory", "GRU", 
          "Gated Recurrent Unit", "BERT", "GPT", "Computer Vision", "CV",
          "Natural Language Processing", "NLP", "Data Science", "Statistics",
          "Regression", "Classification", "Clustering", "Reinforcement Learning", "RL",
          "XGBoost", "LightGBM", "CatBoost", "Random Forest", "Decision Trees",
          "Support Vector Machines", "SVM", "K-Means", "DBSCAN", "Principal Component Analysis",
          "PCA", "t-SNE", "Feature Engineering", "Model Selection", "Cross Validation",
          "Hyperparameter Tuning", "Ensemble Methods", "Gradient Boosting",
          "Neural Architecture Search", "AutoML", "MLOps", "Model Deployment",
          "TensorBoard", "Weights & Biases", "MLflow", "Kubeflow", "Apache Airflow",
          "Jupyter", "Google Colab", "Kaggle", "Time Series Analysis", "A/B Testing"
      ],

      "mobile_development": [
          "Android", "iOS", "Swift", "Objective-C", "Kotlin", "Java", "React Native",
          "Flutter", "Dart", "Xamarin", "Ionic", "Cordova", "PhoneGap",
          "Unity", "Unreal Engine", "Android Studio", "Xcode", "TestFlight",
          "Google Play Console", "App Store Connect", "Firebase", "Realm",
          "SQLite", "Core Data", "Room", "Retrofit", "Alamofire", "Volley",
          "Glide", "Picasso", "Lottie", "Material Design", "Human Interface Guidelines",
          "MVP", "MVVM", "MVI", "Clean Architecture", "Dagger", "Hilt",
          "RxJava", "RxSwift", "Combine", "LiveData", "StateFlow", "Jetpack Compose",
          "SwiftUI", "UIKit", "Storyboard", "Auto Layout", "ConstraintLayout"
      ],

      "devops_tools": [
          "Git", "GitHub", "GitLab", "Bitbucket", "SVN", "Mercurial", "Docker",
          "Kubernetes", "Jenkins", "Travis CI", "CircleCI", "GitLab CI", "GitHub Actions",
          "Ansible", "Puppet", "Chef", "SaltStack", "Terraform", "Pulumi",
          "Vagrant", "Packer", "NGINX", "Apache", "Apache HTTP Server", "Tomcat",
          "Load Balancer", "Reverse Proxy", "Monitoring", "Logging", "Grafana", 
          "Prometheus", "Alertmanager", "Jaeger", "Zipkin", "New Relic", "Datadog",
          "Splunk", "ELK Stack", "Elasticsearch", "Logstash", "Kibana", "Fluentd",
          "Istio", "Linkerd", "Envoy", "Consul", "Vault", "Nomad", "etcd",
          "Zookeeper", "ArgoCD", "Flux", "Helm", "Kustomize", "Skaffold", "Tilt",
          "Buildah", "Podman", "containerd", "CRI-O", "Kaniko", "OpenShift"
      ],

      "testing": [
          "JUnit", "TestNG", "Mockito", "Selenium", "WebDriver", "Cypress", "Playwright",
          "Puppeteer", "Jest", "Mocha", "Chai", "Jasmine", "Karma", "Protractor",
          "Cucumber", "Gherkin", "pytest", "unittest", "nose", "Robot Framework",
          "Postman", "Newman", "REST Assured", "SoapUI", "JMeter", "Gatling",
          "LoadRunner", "K6", "Artillery", "Unit Testing", "Integration Testing",
          "End-to-End Testing", "E2E Testing", "System Testing", "Acceptance Testing",
          "Performance Testing", "Load Testing", "Stress Testing", "Security Testing",
          "Test Automation", "TDD", "Test-Driven Development", "BDD", 
          "Behavior-Driven Development", "ATDD", "Acceptance Test-Driven Development",
          "Continuous Testing", "Test Coverage", "Code Coverage", "Mutation Testing"
      ],

      "data_engineering": [
          "Apache Spark", "Hadoop", "HDFS", "MapReduce", "Apache Kafka", "Apache Storm",
          "Apache Flink", "Apache Beam", "Apache Airflow", "Luigi", "Prefect",
          "Apache NiFi", "Talend", "Informatica", "SSIS", "Azure Data Factory",
          "AWS Glue", "Google Dataflow", "Apache Sqoop", "Apache Flume",
          "Elasticsearch", "Apache Solr", "Apache Lucene", "Redis", "Memcached",
          "Apache Cassandra", "Apache HBase", "Apache Druid", "ClickHouse",
          "Apache Pinot", "Apache Superset", "Tableau", "Power BI", "Looker",
          "Metabase", "Apache Zeppelin", "Jupyter", "Databricks", "Snowflake",
          "BigQuery", "Redshift", "Azure Synapse", "Fivetran", "Stitch", "Singer",
          "dbt", "Great Expectations", "Apache Atlas", "DataHub", "Amundsen"
      ],

      "cybersecurity": [
          "Penetration Testing", "Ethical Hacking", "Vulnerability Assessment", "CISSP",
          "CEH", "CISM", "CISA", "Security+", "Network Security", "Information Security",
          "Cybersecurity", "Firewall", "IDS", "IPS", "SIEM", "SOC", "Incident Response",
          "Malware Analysis", "Reverse Engineering", "Digital Forensics", "Risk Assessment",
          "Compliance", "GDPR", "HIPAA", "SOX", "PCI DSS", "ISO 27001", "NIST",
          "OWASP", "SQL Injection", "XSS", "CSRF", "Buffer Overflow", "Cryptography",
          "PKI", "SSL/TLS", "VPN", "Multi-Factor Authentication", "MFA", "Identity Management",
          "Access Control", "Zero Trust", "Threat Intelligence", "Threat Hunting",
          "Red Team", "Blue Team", "Purple Team", "Metasploit", "Nmap", "Wireshark",
          "Burp Suite", "OWASP ZAP", "Nessus", "OpenVAS", "Qualys", "Rapid7"
      ],

      "blockchain": [
          "Blockchain", "Bitcoin", "Ethereum", "Solidity", "Smart Contracts", "DeFi",
          "NFT", "Web3", "Cryptocurrency", "Crypto", "Hyperledger", "Fabric",
          "R3 Corda", "Ripple", "Stellar", "Cardano", "Polkadot", "Chainlink",
          "IPFS", "Truffle", "Hardhat", "Remix", "MetaMask", "Ganache",
          "OpenZeppelin", "Consensus Algorithms", "Proof of Work", "Proof of Stake",
          "Mining", "Staking", "Yield Farming", "Liquidity Mining", "DEX", "AMM",
          "DAO", "Token Standards", "ERC-20", "ERC-721", "ERC-1155", "BEP-20"
      ],

      "soft_skills": [
          "Leadership", "Communication", "Teamwork", "Team Collaboration", "Problem Solving",
          "Analytical Thinking", "Critical Thinking", "Decision Making", "Project Management",
          "Time Management", "Organization", "Adaptability", "Flexibility", "Creativity",
          "Innovation", "Strategic Thinking", "Planning", "Negotiation", "Conflict Resolution",
          "Emotional Intelligence", "Empathy", "Active Listening", "Public Speaking",
          "Presentation Skills", "Written Communication", "Documentation", "Mentoring",
          "Coaching", "Training", "Customer Service", "Client Relations", "Stakeholder Management",
          "Change Management", "Risk Management", "Quality Assurance", "Process Improvement",
          "Continuous Learning", "Self-Motivation", "Initiative", "Accountability",
          "Reliability", "Attention to Detail", "Multitasking", "Stress Management"
      ],

      "project_management": [
          "Agile", "Scrum", "Kanban", "Lean", "Waterfall", "PRINCE2", "PMI", "PMP",
          "Certified Scrum Master", "CSM", "Product Owner", "JIRA", "Confluence",
          "Trello", "Asana", "Monday.com", "Slack", "Microsoft Teams", "Zoom",
          "Project Planning", "Resource Management", "Budget Management", "Risk Management",
          "Quality Management", "Stakeholder Management", "Change Management",
          "Sprint Planning", "Daily Standups", "Sprint Reviews", "Retrospectives",
          "User Stories", "Acceptance Criteria", "Burndown Charts", "Velocity",
          "Estimation", "Story Points", "Epic", "Feature", "Release Planning"
      ],

      "design_ux_ui": [
          "UI Design", "UX Design", "User Experience", "User Interface", "Figma", "Sketch",
          "Adobe XD", "InVision", "Principle", "Framer", "Zeplin", "Marvel", "Axure",
          "Balsamiq", "Wireframing", "Prototyping", "User Research", "Usability Testing",
          "A/B Testing", "Design Systems", "Style Guide", "Brand Guidelines",
          "Typography", "Color Theory", "Layout Design", "Information Architecture",
          "Interaction Design", "Motion Design", "Animation", "Micro-interactions",
          "Responsive Design", "Mobile Design", "Web Design", "Design Thinking",
          "Human-Centered Design", "Accessibility", "WCAG", "Adobe Creative Suite",
          "Photoshop", "Illustrator", "After Effects", "Cinema 4D", "Blender"
      ],
      "data_science": [
    "Data Analysis", "Data Visualization", "Statistical Analysis", "Predictive Modeling",
    "Data Mining", "Big Data", "ETL", "Data Warehousing", "Business Intelligence",
    "Tableau", "Power BI", "Looker", "Qlik", "D3.js", "SPSS", "SAS", "Stata",
    "RapidMiner", "KNIME", "Alteryx", "Data Cleaning", "Feature Engineering",
    "A/B Testing", "Experimental Design", "Statistical Significance", "Hypothesis Testing",
    "Bayesian Statistics", "Time Series Forecasting", "Regression Analysis",
    "Cluster Analysis", "Factor Analysis", "Conjoint Analysis", "Sentiment Analysis",
    "Network Analysis", "Geospatial Analysis", "Text Mining", "Web Scraping",
    "Data Ethics", "Data Governance", "Data Privacy", "GDPR Compliance"
],

"marketing": [
    "Digital Marketing", "Content Marketing", "SEO", "SEM", "PPC", "Email Marketing",
    "Social Media Marketing", "Influencer Marketing", "Affiliate Marketing",
    "Marketing Automation", "CRM", "Salesforce", "HubSpot", "Marketo", "Pardot",
    "Marketing Analytics", "Google Analytics", "Adobe Analytics", "Mixpanel",
    "Brand Management", "Product Marketing", "Growth Marketing", "Performance Marketing",
    "Marketing Strategy", "Market Research", "Competitive Analysis", "Customer Segmentation",
    "Marketing Campaigns", "Marketing Attribution", "Conversion Rate Optimization",
    "Landing Page Optimization", "A/B Testing", "Marketing ROI", "Marketing Budgeting",
    "Marketing Communications", "Public Relations", "Media Planning", "Media Buying",
    "Copywriting", "Creative Direction", "Marketing Collateral", "Marketing Automation"
],

"sales": [
    "Sales Strategy", "Sales Process", "Sales Methodology", "Solution Selling",
    "Consultative Selling", "Value Selling", "Account Management", "Key Account Management",
    "Sales Operations", "Sales Forecasting", "Sales Pipeline Management", "CRM",
    "Salesforce", "HubSpot Sales", "Zoho CRM", "Microsoft Dynamics", "Lead Generation",
    "Lead Qualification", "Prospecting", "Cold Calling", "Email Outreach", "Social Selling",
    "Sales Presentations", "Product Demonstrations", "Sales Negotiation", "Closing Techniques",
    "Objection Handling", "Sales Training", "Sales Coaching", "Sales Performance Metrics",
    "Quota Attainment", "Sales Analytics", "Sales Reporting", "Territory Management",
    "Channel Sales", "Inside Sales", "Outside Sales", "Enterprise Sales", "SaaS Sales"
],

"human_resources": [
    "Recruitment", "Talent Acquisition", "Sourcing", "Interviewing", "Candidate Assessment",
    "Onboarding", "Employee Relations", "Performance Management", "Employee Engagement",
    "HRIS", "Workday", "BambooHR", "ADP", "Oracle HCM", "Compensation & Benefits",
    "Payroll Administration", "HR Compliance", "Labor Laws", "Employment Law",
    "Diversity & Inclusion", "Employee Training", "Learning & Development",
    "Succession Planning", "Talent Management", "HR Analytics", "Workforce Planning",
    "Organizational Development", "Change Management", "HR Strategy", "Employee Retention",
    "Exit Interviews", "HR Policies", "Employee Handbook", "Disciplinary Actions",
    "Grievance Handling", "Workplace Investigations", "HR Metrics", "HR Reporting"
],



"operations": [
    "Operations Management", "Supply Chain Management", "Logistics", "Inventory Management",
    "Procurement", "Sourcing", "Vendor Management", "Contract Management", "Purchasing",
    "Quality Management", "Six Sigma", "Lean Manufacturing", "Continuous Improvement",
    "Process Optimization", "Business Process Reengineering", "Operations Strategy",
    "Capacity Planning", "Demand Planning", "Production Planning", "Scheduling",
    "Cost Reduction", "Efficiency Improvement", "Operations Research", "Forecasting",
    "Warehouse Management", "Distribution", "Transportation", "Fleet Management",
    "Manufacturing", "Production", "Assembly", "Packaging", "Facilities Management",
    "Maintenance", "Reliability Engineering", "Health & Safety", "Environmental Compliance"
],

"healthcare": [
    "Patient Care", "Clinical Skills", "Diagnosis", "Treatment Planning", "Medical Terminology",
    "Electronic Health Records", "Epic", "Cerner", "Healthcare Compliance", "HIPAA",
    "Medical Coding", "Medical Billing", "Healthcare Analytics", "Population Health",
    "Health Informatics", "Clinical Research", "Clinical Trials", "Regulatory Affairs",
    "Quality Improvement", "Patient Safety", "Risk Management", "Healthcare Administration",
    "Healthcare Management", "Hospital Operations", "Healthcare Policy", "Public Health",
    "Epidemiology", "Biostatistics", "Health Education", "Health Promotion",
    "Mental Health", "Counseling", "Therapy", "Rehabilitation", "Nursing", "Pharmacy",
    "Radiology", "Laboratory", "Surgery", "Emergency Medicine", "Primary Care"
],

"education": [
    "Curriculum Development", "Instructional Design", "Lesson Planning", "Classroom Management",
    "Pedagogy", "Andragogy", "Educational Technology", "Learning Management Systems",
    "Moodle", "Canvas", "Blackboard", "Assessment Design", "Student Evaluation",
    "Educational Psychology", "Learning Theories", "Differentiated Instruction",
    "Inclusive Education", "Special Education", "Gifted Education", "Early Childhood Education",
    "Higher Education", "Adult Education", "Online Teaching", "Distance Learning",
    "Blended Learning", "Educational Leadership", "Academic Advising", "Student Affairs",
    "Educational Research", "Program Evaluation", "Education Policy", "School Administration",
    "Teacher Training", "Professional Development", "Educational Consulting", "Tutoring"
],

"legal": [
    "Legal Research", "Legal Writing", "Contract Law", "Corporate Law", "Litigation",
    "Legal Compliance", "Regulatory Compliance", "Risk Management", "Due Diligence",
    "Intellectual Property", "Patent Law", "Trademark Law", "Copyright Law",
    "Employment Law", "Labor Law", "Real Estate Law", "Environmental Law", "Tax Law",
    "Mergers & Acquisitions", "Securities Law", "Banking Law", "International Law",
    "Legal Technology", "E-Discovery", "Case Management", "Legal Project Management",
    "Legal Strategy", "Negotiation", "Mediation", "Arbitration", "Alternative Dispute Resolution",
    "Legal Analysis", "Legal Drafting", "Client Counseling", "Legal Ethics", "Professional Responsibility"
],

"research": [
    "Research Methodology", "Qualitative Research", "Quantitative Research", "Mixed Methods",
    "Experimental Design", "Survey Design", "Data Collection", "Data Analysis",
    "Statistical Analysis", "Research Ethics", "Literature Review", "Academic Writing",
    "Scientific Writing", "Grant Writing", "Publication", "Peer Review", "Research Presentation",
    "Laboratory Techniques", "Field Research", "Interviews", "Focus Groups", "Observation",
    "Case Studies", "Content Analysis", "Discourse Analysis", "Ethnography", "Action Research",
    "Research Management", "Research Collaboration", "Research Funding", "Research Policy",
    "Research Ethics", "Research Compliance", "Data Management", "Research Software",
    "SPSS", "R", "Python", "NVivo", "Atlas.ti", "EndNote", "Mendeley", "Zotero"
],

"quality_assurance": [
    "Quality Control", "Quality Assurance", "Quality Management Systems", "ISO 9001",
    "Six Sigma", "Lean Six Sigma", "Total Quality Management", "Continuous Improvement",
    "Process Improvement", "Statistical Process Control", "Quality Audits", "Compliance Audits",
    "Root Cause Analysis", "Corrective and Preventive Actions", "Risk Management",
    "Quality Metrics", "Quality Standards", "Quality Planning", "Quality Assurance Testing",
    "Non-Conformance Management", "Supplier Quality", "Vendor Management", "Quality Documentation",
    "Quality Reporting", "Quality Training", "Quality Culture", "Quality Tools",
    "Pareto Analysis", "Fishbone Diagram", "Control Charts", "Failure Mode and Effects Analysis",
    "Quality Cost Analysis", "Quality Assurance Automation", "Test Strategy", "Test Planning"
],

"product_management": [
    "Product Strategy", "Product Roadmap", "Product Lifecycle Management", "Product Development",
    "Product Launch", "Go-to-Market Strategy", "Market Research", "Competitive Analysis",
    "User Research", "Customer Development", "User Stories", "Product Requirements",
    "Prioritization", "Agile Product Management", "Scrum Product Owner", "Kanban",
    "Product Analytics", "Data-Driven Product Management", "A/B Testing", "Feature Adoption",
    "User Engagement", "Retention", "Monetization", "Pricing Strategy", "Revenue Modeling",
    "Product Marketing", "Positioning", "Messaging", "Sales Enablement", "Customer Feedback",
    "Product Metrics", "KPIs", "OKRs", "Product Vision", "Product Leadership",
    "Stakeholder Management", "Cross-Functional Collaboration", "Product-Led Growth"
],

"business_analysis": [
    "Business Requirements", "Functional Requirements", "Non-Functional Requirements",
    "Requirements Elicitation", "Requirements Analysis", "Requirements Documentation",
    "Business Process Modeling", "Process Mapping", "Workflow Analysis", "Gap Analysis",
    "SWOT Analysis", "Feasibility Studies", "Cost-Benefit Analysis", "Business Cases",
    "User Stories", "Use Cases", "Data Modeling", "Data Flow Diagrams", "UML",
    "Business Rules", "System Analysis", "Solution Design", "Solution Evaluation",
    "Stakeholder Analysis", "Change Management", "Risk Assessment", "Business Intelligence",
    "Data Analysis", "Process Improvement", "Business Process Reengineering",
    "Agile Analysis", "Scrum", "Kanban", "BA Tools", "JIRA", "Confluence", "Visio", "Balsamiq"
],

"content_creation": [
    "Content Strategy", "Content Planning", "Content Calendar", "Copywriting", "Technical Writing",
    "Creative Writing", "Editing", "Proofreading", "Content Optimization", "SEO Writing",
    "Content Management Systems", "WordPress", "Drupal", "Joomla", "Content Marketing",
    "Social Media Content", "Blog Writing", "Article Writing", "White Papers", "E-books",
    "Case Studies", "Press Releases", "Email Copy", "Landing Page Copy", "Ad Copy",
    "Video Scripting", "Podcast Scripting", "Storyboarding", "Content Localization",
    "Content Personalization", "Content Distribution", "Content Promotion", "Content Analytics",
    "Content Performance Metrics", "A/B Testing", "Content Audits", "Content Governance",
    "Brand Voice", "Style Guides", "Content Collaboration", "Content Workflow Management"
],

"customer_service": [
    "Customer Support", "Customer Experience", "Customer Success", "Client Relations",
    "Customer Relationship Management", "CRM", "Service Level Agreements", "SLA Management",
    "Customer Satisfaction", "CSAT", "Net Promoter Score", "NPS", "Customer Effort Score",
    "Customer Retention", "Customer Loyalty", "Customer Onboarding", "Customer Training",
    "Technical Support", "Help Desk", "Troubleshooting", "Issue Resolution", "Escalation Management",
    "Customer Communication", "Active Listening", "Empathy", "Conflict Resolution", "De-escalation",
    "Customer Feedback", "Customer Surveys", "Voice of the Customer", "VOC",
    "Service Quality", "Service Recovery", "Customer Service Metrics", "First Contact Resolution",
    "Average Handle Time", "Customer Service Standards", "Customer Service Policies",
    "Multichannel Support", "Omnichannel Support", "Live Chat", "Email Support", "Phone Support"
],
"architecture_engineering": [
    "Architectural Design", "Building Information Modeling", "BIM", "AutoCAD", "Revit",
    "SketchUp", "3D Modeling", "Structural Analysis", "MEP Systems", "Sustainable Design",
    "LEED Certification", "Green Building", "Construction Documentation", "Blueprint Reading",
    "Building Codes", "Zoning Regulations", "Site Planning", "Urban Planning", "Landscape Architecture",
    "Interior Design", "Space Planning", "Materials Science", "Structural Engineering",
    "Civil Engineering", "Mechanical Engineering", "Electrical Engineering", "Environmental Engineering",
    "Project Management", "Construction Management", "Cost Estimation", "Value Engineering",
    "Feasibility Studies", "Building Simulation", "Energy Modeling", "Daylighting Analysis",
    "Acoustics", "Fire Protection", "Accessibility Design", "Universal Design", "Historic Preservation",
    "Facilities Management", "Building Commissioning", "Construction Administration", "Contract Administration"
],

"arts_entertainment": [
    "Acting", "Directing", "Producing", "Screenwriting", "Playwriting", "Cinematography",
    "Film Editing", "Sound Design", "Music Composition", "Songwriting", "Music Production",
    "Audio Engineering", "Mixing", "Mastering", "Live Sound", "Stage Management",
    "Set Design", "Lighting Design", "Costume Design", "Makeup Artistry", "Special Effects",
    "Visual Effects", "VFX", "Animation", "3D Animation", "Motion Graphics", "Graphic Design",
    "Illustration", "Digital Art", "Photography", "Videography", "Choreography", "Dance",
    "Theater Production", "Event Production", "Talent Management", "Artist Representation",
    "Arts Administration", "Curating", "Gallery Management", "Art Direction", "Creative Direction",
    "Storytelling", "Narrative Design", "Character Development", "World Building", "Improv",
    "Voice Acting", "Puppetry", "Magic", "Circus Arts", "Game Design", "Level Design"
],

"construction_trades": [
    "Carpentry", "Framing", "Finish Carpentry", "Cabinet Making", "Welding", "Metal Fabrication",
    "Electrical Wiring", "Electrical Systems", "Plumbing", "Pipefitting", "HVAC Installation",
    "HVAC Repair", "Masonry", "Concrete Work", "Tiling", "Flooring Installation", "Roofing",
    "Siding", "Drywall Installation", "Painting", "Insulation Installation", "Glazing",
    "Heavy Equipment Operation", "Excavation", "Demolition", "Scaffolding", "Rigging",
    "Blueprint Reading", "Construction Math", "Measurement", "Power Tools", "Hand Tools",
    "Safety Protocols", "OSHA Standards", "First Aid", "CPR", "Forklift Operation",
    "Crane Operation", "Welding Certification", "Electrical Licensing", "Plumbing Certification",
    "Project Scheduling", "Quality Control", "Site Supervision", "Materials Handling", "Waste Management"
],

"environmental_science": [
    "Environmental Assessment", "Environmental Impact Analysis", "EIA", "Ecological Surveys",
    "Habitat Restoration", "Wetland Delineation", "Endangered Species Management", "Conservation Biology",
    "Environmental Policy", "Environmental Law", "Sustainability", "Renewable Energy",
    "Solar Energy Systems", "Wind Energy Systems", "Energy Efficiency", "Carbon Footprint Analysis",
    "Climate Science", "Climate Modeling", "Meteorology", "Hydrology", "Water Quality Management",
    "Wastewater Treatment", "Air Quality Management", "Pollution Control", "Waste Management",
    "Recycling Systems", "Composting", "Environmental Monitoring", "Data Collection", "Field Sampling",
    "GIS", "Remote Sensing", "Environmental Modeling", "Risk Assessment", "Remediation Technologies",
    "Soil Science", "Geology", "Environmental Chemistry", "Environmental Toxicology", "Environmental Education"
],

"event_management": [
    "Event Planning", "Event Coordination", "Event Production", "Venue Management", "Vendor Management",
    "Budget Management", "Event Marketing", "Event Promotion", "Sponsorship Acquisition", "Ticketing",
    "Registration Management", "Attendee Management", "Event Technology", "Event Apps", "AV Production",
    "Lighting Design", "Sound Systems", "Stage Management", "Logistics Planning", "Transportation Coordination",
    "Accommodation Arrangements", "Catering Management", "Menu Planning", "Beverage Service",
    "Event Design", "Theme Development", "Decor", "Floral Design", "Entertainment Booking",
    "Speaker Management", "Program Development", "Scheduling", "Timeline Management", "Risk Management",
    "Emergency Planning", "Crowd Management", "Security Planning", "Post-Event Evaluation",
    "Event Analytics", "ROI Measurement", "Client Relations", "Contract Negotiation", "Permitting"
],

"fashion_apparel": [
    "Fashion Design", "Apparel Design", "Textile Design", "Pattern Making", "Draping", "Sewing",
    "Garment Construction", "Tailoring", "Alterations", "Fashion Illustration", "Technical Design",
    "CAD for Fashion", "Adobe Illustrator", "Adobe Photoshop", "Fashion Merchandising", "Retail Buying",
    "Visual Merchandising", "Store Design", "Fashion Marketing", "Brand Management", "Fashion PR",
    "Trend Forecasting", "Color Theory", "Material Sourcing", "Fabric Selection", "Supply Chain Management",
    "Production Management", "Quality Control", "Costing", "Fit Analysis", "Size Grading",
    "Fashion Styling", "Personal Styling", "Editorial Styling", "Costume Design", "Fashion Journalism",
    "Fashion Photography", "Runway Production", "Fashion Show Coordination", "Sustainable Fashion",
    "Ethical Sourcing", "Circular Fashion", "Fashion Technology", "Wearable Technology", "3D Fashion Design"
],

"food_beverage": [
    "Culinary Arts", "Cooking Techniques", "Baking", "Pastry Arts", "Food Preparation", "Food Safety",
    "HACCP", "Sanitation Standards", "Menu Development", "Recipe Development", "Food Costing",
    "Inventory Management", "Kitchen Management", "Catering", "Event Catering", "Restaurant Management",
    "Front of House", "Back of House", "Bartending", "Mixology", "Wine Service", "Sommelier",
    "Beverage Program Development", "Coffee Brewing", "Barista Skills", "Food Styling", "Food Photography",
    "Nutrition", "Dietary Restrictions", "Allergen Management", "Sustainable Sourcing", "Farm-to-Table",
    "Food Science", "Food Technology", "Food Preservation", "Fermentation", "Molecular Gastronomy",
    "Food Writing", "Food Criticism", "Menu Engineering", "Profitability Analysis", "Customer Service",
    "Staff Training", "Health Department Compliance", "Liquor Licensing", "Purchasing", "Vendor Relations"
],

"government_public_admin": [
    "Public Policy", "Policy Analysis", "Policy Development", "Legislative Process", "Regulatory Affairs",
    "Government Relations", "Lobbying", "Public Administration", "Public Finance", "Budget Administration",
    "Grant Writing", "Grant Management", "Public Procurement", "Contract Management", "Compliance Monitoring",
    "Program Evaluation", "Performance Measurement", "Strategic Planning", "Organizational Development",
    "Public Relations", "Government Communications", "Crisis Communications", "Community Engagement",
    "Constituent Services", "Economic Development", "Urban Planning", "Zoning Administration", "Permitting",
    "Licensing", "Inspection Services", "Law Enforcement", "Emergency Management", "Homeland Security",
    "Public Health Administration", "Social Services Administration", "Public Records Management",
    "Freedom of Information Act", "FOIA", "Ethics Compliance", "Government Technology", "E-Government"
],

"hospitality_tourism": [
    "Hotel Management", "Resort Management", "Lodging Operations", "Front Office Operations", "Housekeeping",
    "Concierge Services", "Guest Services", "Revenue Management", "Yield Management", "Hospitality Marketing",
    "Tourism Marketing", "Destination Marketing", "Travel Agency Operations", "Tour Guiding", "Tour Operations",
    "Event Management", "Conference Services", "Banquet Operations", "Catering Management", "Restaurant Management",
    "Food and Beverage Service", "Bar Management", "Spa Management", "Recreation Management", "Golf Operations",
    "Casino Operations", "Cruise Line Operations", "Airline Services", "Rental Car Management", "Tourism Development",
    "Sustainable Tourism", "Ecotourism", "Cultural Tourism", "Adventure Tourism", "Hospitality Technology",
    "Property Management Systems", "PMS", "Global Distribution Systems", "GDS", "Customer Relationship Management",
    "CRM", "Quality Assurance", "Standards Compliance", "Safety Management", "Security Procedures"
],

"insurance": [
    "Underwriting", "Risk Assessment", "Risk Management", "Actuarial Science", "Claims Processing",
    "Claims Investigation", "Claims Adjustment", "Loss Control", "Insurance Sales", "Agency Management",
    "Brokerage Operations", "Reinsurance", "Insurance Regulation", "Compliance", "Policy Administration",
    "Policy Development", "Product Development", "Pricing Strategy", "Premium Auditing", "Insurance Analytics",
    "Fraud Detection", "Subrogation", "Litigation Management", "Insurance Law", "Insurance Contracts",
    "Life Insurance", "Health Insurance", "Property and Casualty", "P&C", "Auto Insurance", "Homeowners Insurance",
    "Commercial Insurance", "Liability Insurance", "Workers' Compensation", "Disability Insurance", "Long-Term Care",
    "Annuities", "Retirement Planning", "Employee Benefits", "Group Benefits", "Insurance Technology",
    "Insurtech", "Customer Service", "Relationship Management", "Renewals Management", "Agency Operations"
],

"manufacturing": [
    "Manufacturing Processes", "Production Planning", "Production Scheduling", "Manufacturing Operations",
    "Lean Manufacturing", "Six Sigma", "Continuous Improvement", "Quality Control", "Quality Assurance",
    "Total Quality Management", "Statistical Process Control", "Process Optimization", "Automation",
    "Robotics", "CNC Machining", "CNC Programming", "Welding", "Metal Fabrication", "Assembly",
    "Packaging", "Materials Management", "Inventory Control", "Supply Chain Management", "Logistics",
    "Maintenance", "Preventive Maintenance", "Predictive Maintenance", "Facility Management",
    "Manufacturing Engineering", "Industrial Engineering", "Process Engineering", "Product Development",
    "New Product Introduction", "NPI", "Manufacturing Technology", "CAD/CAM", "PLM", "ERP Systems",
    "MES", "Manufacturing Execution Systems", "Safety Management", "OSHA Compliance", "Environmental Compliance",
    "Cost Reduction", "Efficiency Improvement", "Capacity Planning", "Workforce Management", "Supervision"
],

"media_communications": [
    "Journalism", "News Writing", "Feature Writing", "Copy Editing", "Proofreading", "Fact-Checking",
    "Broadcast Journalism", "TV Production", "Radio Production", "Podcast Production", "Video Production",
    "Documentary Filmmaking", "Social Media Management", "Content Strategy", "Content Creation",
    "Digital Marketing", "Public Relations", "Media Relations", "Crisis Communications", "Brand Communications",
    "Internal Communications", "Corporate Communications", "Speechwriting", "Press Release Writing",
    "Media Planning", "Media Buying", "Advertising", "Marketing Communications", "Integrated Marketing Communications",
    "Graphic Design", "Web Design", "UX/UI Design", "Photography", "Videography", "Animation",
    "Media Law", "Ethics in Media", "Media Research", "Audience Analysis", "Media Analytics",
    "SEO", "SEM", "Email Marketing", "Social Media Advertising", "Content Management Systems", "CMS"
],


"real_estate": [
    "Real Estate Sales", "Real Estate Brokerage", "Property Management", "Leasing", "Tenant Relations",
    "Rental Property Management", "Commercial Real Estate", "Residential Real Estate", "Real Estate Investment",
    "Real Estate Development", "Property Development", "Land Acquisition", "Zoning", "Land Use Planning",
    "Real Estate Finance", "Mortgage Lending", "Loan Origination", "Underwriting", "Real Estate Appraisal",
    "Property Valuation", "Market Analysis", "Comparative Market Analysis", "CMA", "Real Estate Marketing",
    "Real Estate Advertising", "Open Houses", "Home Staging", "Real Estate Law", "Contracts",
    "Negotiation", "Closing Processes", "Title Insurance", "Escrow", "Real Estate Technology",
    "Proptech", "Real Estate CRM", "Real Estate Analytics", "Investment Analysis", "Cash Flow Analysis",
    "Property Inspection", "Maintenance Coordination", "Vendor Management", "Real Estate Investing",
    "REITs", "Real Estate Syndication", "Crowdfunding", "Sustainable Real Estate", "Green Building"
],

"transportation_logistics": [
    "Transportation Management", "Logistics Coordination", "Supply Chain Management", "Fleet Management",
    "Routing", "Scheduling", "Dispatching", "Freight Brokerage", "Freight Forwarding", "Customs Brokerage",
    "Import/Export Operations", "Trade Compliance", "Customs Clearance", "Warehousing", "Distribution",
    "Inventory Management", "Order Fulfillment", "Last-Mile Delivery", "Drone Operations", "Autonomous Vehicles",
    "Transportation Planning", "Traffic Engineering", "Public Transportation", "Rail Operations", "Aviation",
    "Air Traffic Control", "Ground Support", "Maritime Operations", "Port Operations", "Shipping",
    "Intermodal Transportation", "Cold Chain Logistics", "Hazardous Materials Transportation", "Hazmat",
    "Transportation Safety", "DOT Compliance", "FMCSA Regulations", "Transportation Technology",
    "TMS", "Transportation Management Systems", "GPS Tracking", "Telematics", "Route Optimization",
    "Carrier Management", "Freight Audit", "Cost Control", "Logistics Analytics", "Performance Metrics"
]
        }
        
        self.skill_categories["spoken_languages"] = [
            "English", "Arabic", "French", "German", "Spanish", "Italian", "Portuguese",
            "Russian", "Hindi", "Urdu", "Bengali", "Punjabi", "Gujarati", "Tamil", "Telugu",
            "Malayalam", "Kannada", "Marathi", "Chinese", "Mandarin", "Cantonese", "Japanese",
            "Korean", "Turkish", "Dutch", "Swedish", "Norwegian", "Danish", "Finnish",
            "Greek", "Polish", "Czech", "Slovak", "Romanian", "Hungarian", "Thai",
            "Indonesian", "Malay", "Filipino", "Tagalog", "Vietnamese", "Persian", "Farsi",
            "Pashto", "Hebrew", "Swahili", "Afrikaans"
        ]

        # Flat list + maps
        self.all_skills = []
        self.skill_to_category = {}
        for category, skills in self.skill_categories.items():
            self.all_skills.extend(skills)
            for skill in skills:
                self.skill_to_category[skill.lower()] = category
        self.skills_lowercase = {skill.lower(): skill for skill in self.all_skills}

        self.ambiguous_singletons = {
            "planning", "observation", "innovation", "compliance",
            "licensing", "maintenance", "monitoring", "leadership",
            "communication", "teamwork", "documentation", "negotiation",
            "presentation", "training", "coaching", "risk", "quality",
            "research", "analysis", "testing", "management"
        }

        self.alias_map = {
            "ise": "Cisco ISE",
            "ironport": "Cisco IronPort",
            "esa": "Cisco ESA",
            "panorama": "Panorama",
            "fortimanager": "FortiManager",
            "bigip": "BigIP",
            "ltm": "F5 LTM",
            "waf": "WAF",
            "ngfw": "NGFW",
            "f5": "BigIP",
            "dnac": "Cisco Prime/DNAC",
            "prime": "NGINX",
            "dotnet": ".NET",
            "node": "Node.js",
            "reactjs": "React.js",
            "angularjs": "AngularJS",
            "vuejs": "Vue.js"
        }

        print(f"Skill library initialized with {len(self.all_skills)} total skills across {len(self.skill_categories)} categories")

    def get_all_skills(self):
        return self.all_skills

    def get_skills_by_category(self, category: str):
        return self.skill_categories.get(category, [])

    def get_skill_category(self, skill: str):
        return self.skill_to_category.get(skill.lower())

    def is_ambiguous(self, token: str) -> bool:
        return token.strip().lower() in self.ambiguous_singletons

    def normalize_alias(self, token: str) -> str | None:
        t = token.strip().lower()
        if t in self.skills_lowercase:
            return self.skills_lowercase[t]
        alias = self.alias_map.get(t)
        return self.skills_lowercase.get(alias.lower(), alias) if alias else None

    def find_matching_skills(self, text_skills: list) -> list:
        """
        Match a list of *candidate* skill strings to canonical library entries.
        Only inputs that already look like skills should be passed here.
        """
        from rapidfuzz import process, fuzz

        matched = []
        for raw in text_skills:
            if not raw:
                continue
            s = str(raw).strip()

            direct = self.skills_lowercase.get(s.lower())
            if direct:
                matched.append(direct)
                continue

            alias = self.normalize_alias(s)
            if alias and alias.lower() in self.skills_lowercase:
                matched.append(self.skills_lowercase[alias.lower()])
                continue

            if len(s) >= 4:
                best = process.extractOne(s, self.all_skills, scorer=fuzz.WRatio, score_cutoff=92)
                if best:
                    matched.append(best[0])

        out = sorted(set(matched), key=lambda x: x.lower())
        return out

    def categorize_skills(self, skills: list) -> dict:
        categorized = {}
        for skill in skills:
            cat = self.get_skill_category(skill)
            if not cat:
                continue
            categorized.setdefault(cat, []).append(skill)
        return categorized

    def get_skill_statistics(self) -> dict:
        stats = {category: len(skills) for category, skills in self.skill_categories.items()}
        stats['total'] = len(self.all_skills)
        return stats