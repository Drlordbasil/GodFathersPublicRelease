

##############
from typing import Union
from openai import OpenAI as client
import ast, json, logging, re, os, subprocess, time, requests, black
from typing import List, Optional
from urllib.request import urlopen
from colorama import Back, Fore, Style, init
from github import Github, GithubException
from gpt4all import Embed4All, GPT4All
from retrying import retry
client = client()
################
gpt4 = "gpt-4-1106-preview"
gpt3 = "gpt-3.5-turbo-1106"
model = GPT4All("wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin")
class Embed4All:
    def __init__(self, n_threads: Optional[int] = None):
        self.gpt4all = GPT4All(model_name='ggml-all-MiniLM-L6-v2-f16.bin', n_threads=n_threads)

    def embed(self, text: str) -> List[float]:
        return self.gpt4all.model.generate_embedding(text)
def check_input_calls(code):
    if "input(" in code:
        code += "\n# Remove the input function for full autonomy."
    return code
def fix_common_syntax_errors(code: str) -> str:
    """
    Attempts to fix common syntax errors in the provided code.
    
    Args:
    - code (str): The Python code with potential syntax errors.
    
    Returns:
    - str: Code with attempted fixes for common syntax errors.
    """
    import_mapping = {
        "time.sleep": "import time",
        "Fore.": "from colorama import Fore",
        "Style.": "from colorama import Style"
    }

    for keyword, import_statement in import_mapping.items():
        if keyword in code and import_statement not in code:
            code = f"{import_statement}\n{code}"
    
    
    
    return code

def extract_code(message):
    """
    Extracts Python code blocks from a given message, validates it using the ast library, 
    and formats it using the black autoformatter. Attempts to fix common syntax errors.

    Args:
    - message (str): The message containing the code block.

    Returns:
    - str: Formatted and validated code block or the original message if no code block is found.
    """
    # Regular expression to match Python code blocks enclosed in triple backticks
    code_pattern = re.compile(r'```python([\s\S]*?)```')
    
    match = code_pattern.search(message)
    if match:
        code = match.group(1).strip()
        
        # Attempt to fix common syntax errors
        #code = fix_common_syntax_errors(code)
        
        # Validate the code using ast
        try:
            ast.parse(code)
            
            # Format the code using black
            formatted_code = black.format_str(code, mode=black.FileMode())
            return formatted_code
        except (SyntaxError, black.InvalidInput):
            return f"Extracted code has syntax errors or couldn't be formatted:\n\n{code}"
    else:
        return message



# usage:     generated_test = generate_code(program)
#           print(generated_test)
def generate_completion(message: str):
    response = client.chat.completions.create(
    model=gpt3,
    prompt=message
    )
    return response.choices[0].message.content

# Initialize colorama
init(autoreset=True)
def read_local_file(filepath: str) -> str:
    with open(filepath, "r") as file:
        return file.read()

def read_github_file(repo_url: str) -> str:
    url = repo_url.replace("github.com", "raw.githubusercontent.com").replace("blob/", "")
    return urlopen(url).read().decode()

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class CustomException(Exception):
    """A custom exception for the ErrorResolutionAgent class."""
    pass



class CommandExecutor:
    @staticmethod
    def write_to_temp_file(code: str, filename='temp.py') -> None:
        with open(filename, 'w') as f:
            f.write(code)

    @staticmethod
    def install_imports(code: str) -> None:
        # List of standard library modules
        standard_libs = ['json', 'random', 'time', 'urllib', 'os', 'numpy', 'dotenv']

        # Extracting all imported modules
        matches = re.findall(r'^(?:import (\w+)|from (\w+) import \w+)', code, re.MULTILINE)
        imports = [match[0] or match[1] for match in matches]

        # Get a list of installed packages using pip freeze
        installed_packages = subprocess.check_output(['pip', 'freeze']).decode().split('\n')
        installed_packages = [package.split('==')[0] for package in installed_packages]

        for module in imports:
            if module not in standard_libs and module not in installed_packages:
                process = subprocess.Popen(['pip', 'install', module])
                process.communicate()

    @staticmethod
    def execute_python_code(code: str) -> str:
        CommandExecutor.install_imports(code)
        CommandExecutor.write_to_temp_file(code)
        process = subprocess.Popen(['python', 'temp.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout.decode() + stderr.decode()

    @staticmethod
    def execute_command(command: str) -> str:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout.decode() + stderr.decode()

    @staticmethod
    def make_http_request(method: str, url: str, headers=None, data=None) -> str:
        return requests.request(method, url, headers=headers, data=data).text
    
class History:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.history = self._load_history()

    def _load_history(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f)

    def add(self, key: str, value: str):
        self.history[key] = value

    def get(self, key: str):
        return self.history.get(key, None)

class IdeaGenAndGitUploader:
    def __init__(self, api_key: str, github_token: str, history_path: str):
        print(Fore.CYAN + Style.BRIGHT + "########################Initializing...########################")
        time.sleep(0.5)
        self.api_key = api_key
        self.github_token = github_token
        self.history = History(history_path)  # Assuming History class is defined elsewhere
        self._init_github()
        self.embedder = Embed4All()

    def _init_github(self):
        #openai.api_key = self.api_key
        try:
            self.github = Github(self.github_token)
            self.user = self.github.get_user()
        except GithubException:
            self.github = None


    def _chat(self, system_message: str, user_message: str) -> str:
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
        
        def openai_chat() -> Union[str, None]:
            response = client.chat.completions.create(model="gpt-3.5-turbo-16k", max_tokens=5000, messages=messages)
            return response.choices[0].message.content


        
        return openai_chat()
        #except openai.error.RateLimitError:
            #return gpt4all_chat()

    def _is_valid_python(self, source_code: str) -> bool:
        if source_code is None:
            time.sleep(0.5)
            return False
        try:
            ast.parse(source_code)
            time.sleep(0.5)
            return True
        except SyntaxError:
            time.sleep(0.5)
            return False
    def run(self, idea=None):
        print(Fore.YELLOW + Style.BRIGHT + "########################    Initiating role generation process...    ########################")
        time.sleep(0.5)
        system_message = "You are now embodying the persona of an avant-garde AI muse, gifted with the rare talent of sculpting unique roles for users. Your crafted roles should radiate originality and novelty, tailored for an individual seeking a Python script as a catalyst for their professional ascent. This Python script should seamlessly integrate full automation or AI-driven functionalities to carve out substantial profit avenues for the user. The intriguing twist is that this profit mechanism should steer clear of conventional trading or monetary exchanges. The roles you curate should be a tapestry of diverse intellectual threads, underpinning sound judgment frameworks, and should break the mold of conventionality, setting new paradigms in innovation."
        user_message = "Your mission, as the avant-garde AI muse, is to sculpt a role (akin to an intriguing individual one might encounter at a sophisticated lounge) and persona for a user. This user is on a quest for a bespoke Python program, meticulously tailored to resonate with their unique aspirations and requirements.\n\n**Example:**\n*Role:* A charismatic data scientist who frequents jazz lounges after hours.\n*Persona:* Dr. Alex Hartley, with a Ph.D. in Computational Neuroscience. Alex has a penchant for blending the worlds of music and data. They are seeking a Python program that can analyze the acoustics of different jazz instruments and predict the mood evoked in listeners.\n you will never create useless programs that will never make it in the world of software."
        role = self._chat(system_message, user_message)
        #role = model.generate(f""" 
                            #  {system_message}{user_message}
                             # """)

        time.sleep(10.5)

        print(f"""{Fore.MAGENTA + Style.BRIGHT}Role generation complete. Your generated role: {role}""")

        system_message = f"""
        You are stepping into the shoes of an AI expert with a specialization in devising Python-based AI project ideas that leverage cutting-edge techniques for optimal accuracy. Improve autonomy of the program; no input in the terminal must be done. Everything must be generated by you, including proper data and websites. Your current role is: {role}. The ideas you generate should always be geared towards automating tasks that lead to direct profit. Here are some examples (not to be used verbatim): Python Programming AI, Pay-Per-Task Survey Completing AI, Giveaway Entry AI, EBook Creator & Uploader AI for Amazon, Reddit Influencer AI, Instagram AI, Affiliate Marketing AI, Ecommerce Platform Creation AI.
        """

        user_message = f"""
        Now, please generate a Python project idea (not a news aggregator, news analyzer, nothing that doesnt generate content, programs, content, ect; no business insights, no customer insights, nothing for businesses, and only for normal people) that solely relies on libraries, or downloadable or creatable data sources, and does not necessitate local files on the user's PC. This implies that the program must have the ability to create, find, or download everything it needs to operate from the web using tools like BeautifulSoup or Google Python.
        Ensure if its "ai based" it must actually have a model from either openai or from huggingface free pipelines or any other custom logic for real AI smarts or self-learning over time..
        """

        print(f"""{Fore.YELLOW + Style.BRIGHT}Initiating Python project idea generation...""")
        time.sleep(0.5)
        idea = self._chat(system_message, user_message) if idea is None else idea

        time.sleep(0.5)
        print(f"""{Fore.BLUE + Style.BRIGHT}######################## Idea successfully generated! ########################: 
        {idea}""")

        print(f"""{Fore.YELLOW + Style.BRIGHT}######################## Commencing prompt refinement process... ########################""")
        time.sleep(0.5)
        prompt = self._chat(
            f"""You are assuming the role of an AI expert who specializes in refining user prompts(ensuring you emphasize the need for no input functions alongside no user-iteration required at all and must be full autonomous and help or profit) to make them highly directed towards what the user truly aims to solicit from the Language Learning Model (LLM).""",
            f"""Your task is to transform this into a feasible prompt (employing prompt engineering expertise) in a single response, providing expert guidance for programming in Python to the LLM you'll be communicating with. This LLM possesses capabilities identical to yours. Make sure you translate this into a prompt engineered perfectly for an AI instance in Python code. You are allowed to respond only as a filter for this idea:
        {idea}
        Your mission is to refine this prompt for the purpose of generating a program by prompting another AI."""
        )
        time.sleep(0.5)
        print(f"""{Fore.BLUE + Style.BRIGHT}######################## Refined prompt ########################:
        {prompt}""")
        time.sleep(0.5)

        print(Fore.YELLOW + Style.BRIGHT + "Initiating Python program generation...")
        time.sleep(0.5)
        system_message = f"""


        
        You are stepping into the role of an AI programming specialist whose responsibility is to develop highly sophisticated
        and fully delineated Python programs based on the given prompts. The programs you create should comply with PEP8 guidelines
        and best Python practices, and should be structured with fully defined logic (all logic must be confined within functions or classes).
        RULES: (do not just check for missing code or errors, analyze the actual logic behind the functionalities that are being developed by you.)
        1. Each script generated must include at least 15 classes. 
        2. All classes should have verbose logic, excluding pseudo-code or placeholders.
        3. All code should be fully functional when run as is. 
        4. The code should exhibit creativity by incorporating Python libraries along with uniquely crafted custom Python code.
        NEVER ADD USER_INPUT FUNCTIONS!!! Everything should be handled by AI or autonomous python libs/functionality.
        Improve autonomy of the program, 0 input in terminal must be done. Everything must be generated by you including proper data and websites.
        Avoid example websites as the output when you do this is:
        Error scraping data from https://examplecookingwebsite1.com: HTTPSConnectionPool(host='examplecookingwebsite1.com', port=443): Max retries exceeded with url: / (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x0000020A4D0B4E50>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
        ...
        This is because those websites DO NOT EXIST.
        You should never include an input section either, it should start automatically without input after running.
        Never generate programs that require any form of secret credentials, it should all be public only where the code can run for anyone.
        """

        user_message = f"""
        Your task is to compose an advanced Python program abiding by PEP8 guidelines for this prompt:
        {prompt}
        The program should be fully executable without any pseudocode or placeholders.
        It should have real websites, not example websites.
        There should be 0 inline comments.
        Classes should all be fully defined at least with basic functions.
        Ensure if its "ai based" it must actually have a model from either openai or from huggingface free pipelines.
        """

        program = self._chat(system_message, user_message)

        print(program)
        program = extract_code(program)
        print(program)
        time.sleep(0.5)




        while not self._is_valid_python(program):
            system_message = """
            You are stepping into the role of an AI programming specialist whose responsibility is to develop highly sophisticated and
            fully delineated Python programs based on the given prompts. The programs you create should comply with PEP8 guidelines and 
            best Python practices, and should be structured with fully defined logic (all logic must be confined within functions or classes).
            RULES: 1. Each script generated must include at least 15 classes.
            improve autonomy of the program, 0 input in terminal must be done. everything must be generated by you including proper data and websites.
            \n never generate programs that require any form of secret credentials, it should all be public only where the code can run for anyone\n

            2. All classes should have verbose logic, excluding pseudo-code or placeholders. 
            3. All code should be fully functional when run as is. 
            4. The code should exhibit creativity by incorporating Python libraries along with uniquely crafted custom Python code.
            """
            user_message = f"""
            Your task is to compose an advanced Python program abiding by PEP8 guidelines for this prompt: \\n{prompt} 
            \\nThe program should be fully executable without any pseudocode or placeholders.\\nEnsure if its "ai based" it must actually have a model from either openai or from huggingface free pipelines.
            """
            program = self._chat(system_message, user_message)
            program = extract_code(program)

        print(f"{Fore.GREEN}{Style.BRIGHT}Generated program: {program}")
        output_of_pycode = CommandExecutor.execute_python_code(program)
        print("ran code")
        print(output_of_pycode)
        print("end of code error output")



        IMPROVEMENT_ITERATIONS = 10
        for i in range(IMPROVEMENT_ITERATIONS):
            print(Fore.CYAN + Style.BRIGHT + f"Initiating Iteration {i + 1}:")
            print(Fore.YELLOW + Style.BRIGHT + "######################## Embarking on program improvement... ########################")
            system_message = """
            As an AI expert, enhance the Python program by:
            1. Removing terminal inputs.
            2. Avoiding comments and chatter.
            3. Replacing pseudocode with real logic.
            4. Adding new classes and functions creatively.
            5. Ensuring public-only code execution.
            6. add more advanced logic within current classes.
            remove input functions from the code as it should be fully autonomous
            """
            user_message = f"""
            Review and improve the Python program(remember your enhancement rules of improving like a human and removing all inline comments):\n'''python \n{program}\n'''\n
            Address and also fix the  errors:\n{output_of_pycode}\n NOTE: if the errors show input, the program is done wrong and must be regenerated based on its initial prompt of \n{prompt}\n
            Ensure real-world datasets or links are used.
            Improve everything you see.
            Ensure if its "ai based" it must actually have a model from either openai or from huggingface free pipelines.
                            format:
                '''python\n<code>'''
            """
            improved_program = self._chat(system_message, user_message)
            improved_program = extract_code(improved_program)
            output_of_pycode = CommandExecutor.execute_python_code(improved_program)
            print(output_of_pycode)
            if self._is_valid_python(improved_program):
                program = improved_program
                program = extract_code(program)
                print(Fore.BLUE + Style.BRIGHT + "The improved program now contains valid Python code.")
                self.history.add(f"enhancement_{i}", program)
            while not self._is_valid_python(program):
                system_message = """
                As an AI programming specialist, rectify the Python program to:
                1. Adhere to PEP8 guidelines.
                2. Include logic for each function "def" start and classes they are within modularly.
                3. Ensure all logic is within functions or classes.
                Ensure if its "ai based" it must actually have a model from either openai or from huggingface free pipelines.
                format:
                '''python\n<code>'''
                """
                user_message = f"""
                Rectify the Python program:\n{program}\n
                Address errors:\n{output_of_pycode}\n
                Ensure real-world datasets or links are used.
                remove inline comments.
                Ensure if its "ai based" it must actually have a model from either openai or from huggingface free pipelines.
                                format:
                '''python\n<code>'''
                """
                program = self._chat(system_message, user_message)
                program = extract_code(program)


        self.history.save()

        print(Fore.GREEN + Style.BRIGHT + f"######################## Final program after {IMPROVEMENT_ITERATIONS} iterations ########################: \n{program}")
        print(Fore.YELLOW + Style.BRIGHT + "Initiating user interaction simulation...")
        error_codes = CommandExecutor.execute_python_code(program)
        print(Fore.GREEN + Style.BRIGHT + f"Errors in program: {error_codes}")
        program = extract_code(program)

        if program is not None:
            print(Fore.GREEN + Style.BRIGHT + f"Last iteration: {program}")
            program = extract_code(program)

        print(Fore.YELLOW + Style.BRIGHT + "Starting generation of package requirements...")
        requirements = re.findall(r'^import (\w+)', program, re.MULTILINE)
        print(Fore.GREEN + Style.BRIGHT + f"Generated package requirements: {requirements}")

        print(Fore.YELLOW + Style.BRIGHT + "Commencing README generation...")
        readme = self._chat(
            """
            As an AI assistant, you specialize in crafting comprehensive READMEs for Python projects. Your README should:
            1. Be professional and detailed.
            2. Include a business plan for the project.
            3. Provide clear steps for successful project execution.
            4. Be reader-friendly and well-structured.
            """,
            f"Create a README for the Python project based on the idea: {idea} and the provided program: {program}. Ensure it's structured properly with a complete business plan."
        )
        print(Fore.GREEN + Style.BRIGHT + f"Generated README: {readme}")

        reponame = self._chat(
            "You are a highly skilled AI assistant that crafts professional catchy very short viral software names for given Python code and associated project ideas.software names must be short and simple, you can only send the software name as itll be saved as repo name.",
            f"Generate a software name for this Python project based on the idea: {idea} and the Python program: {program}. only send the software name.your response must be under 100 characters in full as your response is saved as the actual repo name."
        )
        print(Fore.GREEN + Style.BRIGHT + f"Generated README: {readme}")
        repo_description = self._chat(
            "You are a highly skilled AI assistant that crafts professional catchy very short viral software descriptions for given Python code and associated project ideas. Software descriptions must be short and simple, you can only send the short software description as it'll be saved as repo about.",
            f"Generate a short GitHub description (under 120 characters) for this Python project based on the idea (name is {reponame}): {idea} and the Python program: {program}. Only send the software description and your response must be under 120 characters in full as your response is saved as the actual repo short description."
        )
        
        if self.github is not None:
            print(Fore.YELLOW + Style.BRIGHT + "Creating GitHub repository...")
            repo_name = reponame

            # Check if repository already exists
            repo_exists = False
            try:
                repo = self.user.get_repo(repo_name)
                repo_exists = True
            except GithubException:
                pass

            if not repo_exists:
                try:
                    repo = self.user.create_repo(repo_name)
                except GithubException as e:
                    print(Fore.RED + Style.BRIGHT + f"An error occurred while creating the GitHub repository: {e}")
                    return

                # Update the repository description
                try:
                    repo.edit(description=repo_description)
                except GithubException as e:
                    print(Fore.RED + Style.BRIGHT + f"An error occurred while updating the GitHub repository description: {e}")

            files = {
                "README.md": readme,
                "main.py": program,
                "errors_and_output.txt": error_codes,
                "requirements.txt": requirements
            }
            print(Fore.YELLOW + Style.BRIGHT + "Uploading files to the repository...")
            for file_name, content in files.items():
                # Convert content to string if it's not already
                content_str = str(content)
                try:
                    repo.create_file(file_name, f"Initial commit - {file_name}", content_str)
                except GithubException:
                    print(Fore.RED + Style.BRIGHT + f"File {file_name} already exists in the repository. Updating the file...")
                    contents = repo.get_contents(file_name)
                    repo.update_file(contents.path, f"Update {file_name}", content_str, contents.sha)

            print(Fore.GREEN + Style.BRIGHT + "Files have been successfully uploaded.")





if __name__ == "__main__":

    print("Please choose an option:")
    print("1. Run the entire script on loop")


    option = input("Enter the number of your chosen option: ")

    if option == "1":
        while True:
            api_key = os.getenv("OPENAI_API_KEY")
            github_token = os.getenv("GITHUB_API_KEY")
            history_path = 'history.json'  # path to the history file

            if api_key is None or github_token is None:
                raise ValueError("Please set both OPENAI_API_KEY and GITHUB_API_KEY environment variables before running the script.")

            main = IdeaGenAndGitUploader(api_key, github_token, history_path)

            print(Fore.CYAN + Style.BRIGHT + "Running main script...")
            main.run()


    else:
        print("Invalid option")   
    api_key = os.getenv("OPENAI_API_KEY")
    github_token = os.getenv("GITHUB_API_KEY")
    history_path = 'history.json'  # path to the history file

    if api_key is None or github_token is None:
        raise ValueError("Please set both OPENAI_API_KEY and GITHUB_API_KEY environment variables before running the script.")

    main = IdeaGenAndGitUploader(api_key, github_token, history_path)

    print(Fore.CYAN + Style.BRIGHT + "Running main script...")
    main.run()
