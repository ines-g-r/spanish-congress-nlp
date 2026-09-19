"""
Data Collection Pipeline

Extracts, tokenizes, and structures Spanish parliamentary debate transcripts 
from raw PDF documents into standardized JSON datasets.
"""


from sly import Lexer
import re
import json
import os
import pymupdf
import requests
import tempfile
import Levenshtein
import concurrent.futures 
import time
from party_mapper import PartyLoader


class InterventionLexer(Lexer):
    """
    Tokenizes and structures raw Spanish parliamentary text.

    Uses regular expressions to separate speaker declarations (ORATOR),
    spoken dialogue (TEXT), and procedural annotations (PARENTHESES).
    """

    tokens = {ORATOR, TEXT, PARENTHESES, PDFCODE, LASTLINE}
    ignore = ' \t\n'
    
    PDFCODE = r'cve: .+\s'
    ORATOR = r'((La|El|LA|EL)(\s?[^()\n:\.\,\s]+)?\s?(señora|señor|seño|seor)(\s+[^()\n:\.\,\s]+(\s[^()\n:\.\,\s]+)?){0,3}\.?:)|((La|El|LA|EL|DEL|DE LA)\s?(señora|señor|seño|seor|SEÑORA|SEÑOR)([^()\n\.:]+?)(\.[A-ZÁÉÍÓÚÜÑ]){0,4}((\n)?([^()\n:\.]+?)?){1,2}((\(|\[)[^\n:]+(\n[^\n:]*?)?(\)|\])+)\*?:)|((La|El|LA|EL)([^a-z()\n:\.\,]+):)|((La|El|LA|EL)([^()\n\.:]+?)(\n)?([^()\n:]+?)?(\([^()\n:]+(\n[^()\n:]*?)?\)):)|(DISCURSO\?(DEL|DE LA)\s?(SEÑOR|SEÑORA)?\s??([A-ZÁÉÍÓÚÜÑ\s]+?)\.)|((SU MAJESTAD)?\s*(La|El|LA|EL)?\s+(Señora|Señor|SEÑORA|SEÑOR|REY|REINA)\s+(DON|DOÑA)?[^a-z()\n:\.\,]+(:|;))|((La|El|LA|EL)\s?(señora|señor|seño|seor|SEÑORA|SEÑOR)\s?[A-ZÁÉÍÓÚÀÈÌÒÙÑ]+([^()\n\:\.]+)\.?;)|((La|El|LA|EL)\s?(señora|señor|seño|seor|SEÑORA|SEÑOR)([^()\.:]+?)((\n)?([^()\n\.:]+?)?){1,2}(\([^\n:]+(\n[^\n:]*?)?(\n[^\n:]*?)?\)):)' 
    PARENTHESES = r'\([^()]*\)\.?' 
    LASTLINE = r'Se levanta la sesión\.(.*)\n\s*(.*)'
    TEXT = r'[^()\n]+'

    def __init__(self):
        self.orator_num = 0
        self.text_num = 0
        self.parenthesis_num = 0    
        self.orator_name1 = None
        self.orator_name2 = None
        self.orator1, self.orator2 = None, None
        self.special_orator_name = None
        self.texts = {}
        self.orators = {}
        self.new_tuple = ()
        self.gender = None

    def get_orator(self, name: str) -> str:
    
        if name not in self.orators.keys():
            self.orators[name] = 0
        
        self.orators[name] += 1
        self.orator = name+" "+str(self.orators[name])
        
        return self.orator
    
    def ORATOR(self,t):
        genderw = re.search(r"señora|señor|rey|reina|princesa|príncipe", t.value, flags=re.IGNORECASE)
        if genderw:
            word = genderw.group().lower()
            if word in ("señor","rey", "príncipe"):
                self.gender = "M"
            elif word in ("señora", "reina", "princesa"):
                self.gender = "F"
        else:
            genderw = re.search(r"La|El", t.value, flags=re.IGNORECASE)
            if genderw:
                word = genderw.group().lower()
                if word == "el":
                    self.gender = "M"
                elif word == "la":
                    self.gender = "F"

        self.orator_num += 1
        name = re.sub(r'\(Realiza[^():]+\)', '', t.value).strip()
        name = re.sub(r'^(SU MAJESTAD)?(La|El)?\s*(señora|señor|seño|seor|rey|reina|príncipe|princesa)?\s*(don|doña)?', '', name.strip(':').replace('*', ''), flags=re.IGNORECASE)
        name = name.replace('\n', '')
        name = name.strip('. ')
        name = name.strip(';')
        name = name.replace('?', '')
        name = re.sub(r'^(Discurso\s?(del|de la)\s??(señor|señora)?)\s?', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[\xad]', '', name)
        name = re.sub(r'[\xa0]', '', name)  
        name = re.sub(r"[\u2010-\u2015]", "-", name)
        name = re.sub(r'(?<=[A-ZÁÉÍÓÚÀÈÌÒÙÑÇ])\d', '', name)
        name = re.sub(r'(\(((?:[\d\-\_]+)|(([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ\-\.\,\_]){1,10})([a-záéíóúàèìòùñç]{0,2}){1,2})(\)(?!$)))|(\(([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ\-\.\,\_\s]){1,50}(\)(?!$))\s*(,|(?=\()))', '', name)
        self.orator_name1 = re.sub(r'\n', '', name).strip()

        print("ORADORRR", self.orator_name1)
      
        start = min((i for i in (self.orator_name1.find("("), self.orator_name1.find("[")) if i != -1), default=-1)
        end = max((i for i in (self.orator_name1.rfind(")"), self.orator_name1.rfind("]")) if i != -1), default=-1)

        if start != -1 and end !=-1 and end > start:
            self.orator_name2 = self.orator_name1[:start]
            self.orator_name2 = self.orator_name2.upper().strip()
            if self.orator_name2 not in self.orators:
                self.orator2 = self.get_orator(self.orator_name2)
                self.texts[self.orator2] = []
            self.orator_name1 = self.orator_name1[start+1:end]
            self.orator_name1 = self.orator_name1.upper().strip()
            print("ORADOR1", self.orator_name1)
            print("ORADOR2", self.orator_name2)

            self.new_tuple = (self.orator_name1, self.orator_name2)
       
        self.orator1 = self.get_orator(self.orator_name1)
        self.texts[self.orator1] = []
     
        return t

    def PARENTHESES(self,t):
        self.parenthesis_num += 1
        return t
        
    def TEXT(self, t):
        t.value = re.sub(r'\xad', ' ', t.value)
        t.value = re.sub(r'\xa0', ' ', t.value)
        t.value = re.sub(r'\u2000', ' ', t.value)
        t.value = re.sub(r'\u2001', ' ', t.value)
        t.value = re.sub(r'\u2002', ' ', t.value)
        t.value = re.sub(r'\u2003', ' ', t.value)
        t.value = re.sub(r'\u2004', ' ', t.value)
        t.value = re.sub(r'\u2005', ' ', t.value)
        t.value = re.sub(r'\u2006', ' ', t.value)
        t.value = re.sub(r'\u2007', ' ', t.value)
        t.value = re.sub(r'\u2008', ' ', t.value)
        t.value = re.sub(r'\u2009', ' ', t.value)
        t.value = re.sub(r'\u200a', ' ', t.value)
        t.value = re.sub(r'\u200b', ' ', t.value)
        t.value = re.sub(r'\u200c', ' ', t.value)
        t.value = re.sub(r'\u200d', ' ', t.value)
        t.value = re.sub(r'\u200e', ' ', t.value)
        t.value = re.sub(r'\u202f', ' ', t.value)
        t.value = re.sub(r'\u205f', ' ', t.value)
        t.value = re.sub(r'\u2060', ' ', t.value)
        t.value = re.sub(r'\u3000', ' ', t.value)
        t.value = re.sub(r'\ufeff', ' ', t.value)

        t.value = t.value.strip()
        t.value = re.sub(r'((Muchas|Muchísimas|Moitas|Moltes)(\s|\n)+?)?(Mila(\s|\n)+?esker|Gracias|Grazas|Gràcies|Eskerrik(\s|\n)+?asko)[^\.]*\.*((\s|\n)*?)?', '', t.value, flags=re.IGNORECASE)
        t.value = re.sub(r'(Buenos(\s|\n)+?días|Bos(\s|\n)+?días|Bons(\s|\n)+?dies|Egun(\s|\n)+?on)([^\.]+?){1,6}?(\s*?)?\.(\s*?)?', '', t.value, flags=re.IGNORECASE)
        t.value = re.sub(r'(Buenas(\s|\n)+?tardes|Boas(\s|\n)+?tardes|Bona(\s|\n)+?tarda|Arratsalde(\s|\n)+?on)([^\.]+?){1,6}?(\s*?)?\.(\s*?)?', '', t.value, flags=re.IGNORECASE)
        
        t.value = t.value.strip()
        
        if self.orator1:
            self.texts[self.orator1].append(t.value + " ")
            self.text_num += 1
        if self.orator2:
            self.texts[self.orator2].append(t.value + " ")
            self.text_num += 1
      
        return t
    
    def error(self, t):
        print(f"Illegal character {t.value[0]} at index {self.index}")
        self.index += 1

    def print_output(self, orator_name: str, special_orators: dict) -> tuple[str, dict, str | None]:    
        to_delete = []
        for key, value in self.texts.items():
        
            if self.new_tuple:
         
                if key == orator_name.upper():
                    print("")
                    special_orators[self.new_tuple[0]] = self.new_tuple[1]
            if len(value) < 2:
                to_delete.append(key)
                        
        for key in reversed(list(to_delete)):
            base_key = re.sub(r'\d+$', '', key).strip()
            index = int(key.replace(base_key, ''))
            
            del self.texts[key]
            to_delete.remove(key)

            self.orators[base_key] -= 1
            moved = False
            while f"{base_key} {index + 1}" in self.texts:
                next_key = f"{base_key} {index + 1}"
                self.texts[f"{base_key} {index}"] = self.texts[next_key]
                index += 1
                moved = True

            if moved:
                del self.texts[f"{base_key} {index}"]
                
        to_delete = []
        result = "".join(["".join(value) for key, value in self.texts.items() if (len(key[:-1].split())==1 and re.search(re.escape(key[:-1]) + r".*" + re.escape(key[-1:]), orator_name.upper()) != None)
                                                                            or (len(orator_name[:-1].split())==1 and re.search(re.escape(orator_name[:-1].upper()) + r".*" + re.escape(orator_name[-1:]), key) != None)
                                                                            or (len(key[:-1].split())>1 and re.search(re.escape(" ".join(key[:-1].split()[1:])) + r".*" + re.escape(key[-1:]), orator_name.upper()) != None)
                                                                            or (Levenshtein.ratio(key, orator_name.upper()) >= 0.8 and key[-1]==orator_name.upper()[-1])])
        if not result:
            result = "No disponible"

        return result, special_orators, self.gender



class DataCollector():
    """
    Handles PDF downloading, page clipping, text extraction,
    and exporting structured speaker data to JSON datasets.
    """

    def __init__(self):
        self.errors = []
        self.errors_num = 0
        self.pdf_nopage = 0
        self.invalid_url_num = 0
        self.pdf_noorator = 0
        self.pdf_notext = 0
      
    def read_files(self, folder_path: str) -> tuple[list, int]:

        if not os.path.exists(folder_path):
            print(f"La carpeta '{folder_path}' no existe.")

        else:
            data = []
            seen = set()
            esquema = {
                "FASE": "fase",
                "LEGISLATURA": "legislatura",
                "OBJETOINICIATIVA": "objeto_iniciativa",
                "TIPOINICIATIVA": "tipo",
                "ORADOR": "orador",
                "SESION": "fecha",
                "INICIOINTERVENCION": "hora_inicio",
                "FININTERVENCION": "hora_fin",
                "ORGANO": "nombre_sesion",
                "ENLACEPDF": "enlace_pdf"
                }
            for folder_path, _, files in os.walk(folder_path):
                for file_name in files:
                    if file_name.endswith('.json'):
                        json_path = os.path.join(folder_path, file_name)
                        with open(json_path, 'r', encoding='utf-8') as file:
                            try:
                                json_data = json.load(file)
                                for dic in json_data:
                                    for def_key, possible_key in esquema.items():
                                        if possible_key in dic:
                                            dic[def_key] = dic.pop(possible_key)
                                    pdf_id, pdf_page = self.get_page_and_id(dic)

                                    if "ORADOR" in dic:
                                        orator = re.findall(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñÀÈÌÒÙàèìòù\'()]+', re.sub(r'\(.*\)', "", dic["ORADOR"]))
                                        if "INICIOINTERVENCION" in dic:
                                            intv_t = dic["INICIOINTERVENCION"][:5]
                                            dic["INICIOINTERVENCION"] = intv_t
                                            if "FININTERVENCION" in dic:
                                                dic["FININTERVENCION"] = dic["FININTERVENCION"][:5]
                                        elif "FININTERVENCION" in dic:
                                            intv_t = dic["FININTERVENCION"][:5]
                                        else:
                                            
                                            intv_t = None

                                        intervention_id = f"{pdf_id}{pdf_page}{''.join(orator)}{intv_t}"
                                        if intervention_id not in seen:
                                            data.append(dic)
                                            if intv_t != None:
                                                seen.add(intervention_id)
                                    else:
                                        print(f"Error al procesar la intervención: Intervención sin orador")
                                        self.pdf_noorator += 1
                                            
                            except json.JSONDecodeError:
                                print(f"Error al leer el archivo '{file_name}'.")
                print(f"Datos leidos de {folder_path}")
        return data, self.pdf_noorator
        

    def process_pdf(self, doc: pymupdf.Document, dic: dict, pdf_path: str) -> str:
        """
        Extracts speech text from specified PDF pages.

        Args:
            doc: Open PyMuPDF document object.
            dic: Metadata dictionary for a single speech intervention (session, date, speaker, etc.).
            pdf_path: Source URL containing target page number ('page=N').

        Returns:
            str: Combined raw text extracted across relevant PDF pages.

        """

        complete_text = ""
        if "page=" in pdf_path:
            page_num = int(pdf_path.split('page=')[1]) - 1
        else: page_num = None
        
        if page_num != None:
            while page_num < len(doc):
                page = doc[page_num]  
                rect = page.rect
                height = 100
                clip = pymupdf.Rect(0, height, rect.width, rect.height -50)
                text = page.get_text(clip=clip)
                complete_text += text 
                if (page_num + 1) < len(doc):
                    next_page = doc[page_num+1]
                    next_page_text = next_page.get_text(clip=clip) 
                    next_page_text = re.search(r'^(.*?)(?=((La señora|El señor)([^()\n]+)\:)|((La señora|El señor)([^()\n]+?)(\n)?([^()\n]+?)?(\([^()\n]+(\n[^()\n]*?)?\))\:))', next_page_text, re.DOTALL)
                    if next_page_text:
                        complete_text += next_page_text.group()
                        break

                page_num += 1

        else: 
            print("Error al procesar la intervención: URL del PDF sin página indicada")
            self.errors.append(("Error - URL del PDF sin página indicada", dic["SESION"], pdf_path))
            self.pdf_nopage += 1  
        
        return complete_text
    

    def get_text(self, orator_name: str, lexer: InterventionLexer, dic: dict, intervention_id: str, processed_interventions: dict, special_orators: dict) -> tuple[str, dict, str | None]:

        intervention_text, special_orators, gender = lexer.print_output(orator_name+str(processed_interventions[intervention_id]), special_orators) 
  
        if intervention_text == "No disponible":
            if orator_name.upper() in special_orators:
                orator_name = special_orators[orator_name.upper()]
                intervention_text, special_orators,gender = lexer.print_output(orator_name+str(processed_interventions[intervention_id]), special_orators)  
        
        elif intervention_text == "No disponible":
            orator_name = ''.join(dic["CARGOORADOR"].split(' '))
            intervention_text, special_orators,gender = lexer.print_output(orator_name+str(processed_interventions[intervention_id]), special_orators)  
        
        return intervention_text, special_orators, gender
    

    def process_intervention(self, dic: dict, processed_interventions: dict, special_orators: dict, last_pdf: str|None, distinct_leg: bool) -> tuple:
        """
        Fetches a remote PDF transcript, parses speech content, and maps text to the speaker.
        """
        
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        intervention_text, gender = None, None
        pdf_path = (dic["ENLACEPDF"])
        
        try:
            response = requests.get(pdf_path, stream=True, headers=headers)

            if response.status_code == 200: 
                if 'application/pdf' in response.headers.get('Content-Type', ''):
                    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                    tmp.write(response.content)
                            
                    try:
                        doc = pymupdf.open(tmp)

                        complete_text = self.process_pdf(doc, dic, pdf_path)

                        lexer = InterventionLexer()
                        list(lexer.tokenize(complete_text))
                        
                        pdf_id, pdf_page = self.get_page_and_id(dic)
                        
                        if "ORADOR" in dic:
                            orator_name = dic["ORADOR"].split(',')[0].strip()
                            orator = re.findall(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñÀÈÌÒÙàèìòù\'()]+', re.sub(r'\(.*\)', "", dic["ORADOR"]))
                          
                            intervention_id = f"{pdf_id}{pdf_page}{''.join(orator)}"
                           
                            if intervention_id not in processed_interventions:
                                processed_interventions[intervention_id] = 0
                            processed_interventions[intervention_id] += 1
                         
                            intervention_text, special_orators, gender = self.get_text(orator_name, lexer, dic, intervention_id, processed_interventions, special_orators) 
                            if intervention_text == "No disponible":
                                self.errors.append(("Texto no disponible", dic["SESION"], dic["ORADOR"], pdf_path))
                                self.pdf_notext += 1
                            if last_pdf != pdf_id:
                                distinct_leg = True
                            last_pdf = pdf_id

                        if len(processed_interventions)>=2 and not any(pdf_id in k for k in (list(processed_interventions.keys())[-4:])):
                            processed_interventions = {}
                        
                    except Exception as e:
                        print(f"Error al procesar la intervención: {e}")
                        self.errors.append(("Error - Orador no reconocido en el PDF", dic["SESION"], dic["ORADOR"], pdf_path))
                        self.errors_num += 1
                        
                
                    finally:
                        doc.close()

            else:
                print("El contenido no es PDF.")

        except requests.exceptions.RequestException as e:
            print(f"Error al obtener el PDF: {e}")
            self.errors.append(("Error - Obtención del PDF fallida", dic["SESION"], pdf_path))
            self.invalid_url_num += 1
        
        return intervention_text, self.pdf_nopage, self.errors_num, self.invalid_url_num, self.pdf_notext, self.errors, processed_interventions, special_orators, last_pdf, distinct_leg, gender
    

    def create_output_folder(self, dic: dict) -> str:
        dic["ORGANO"] = dic["ORGANO"].replace('\n', ' ')
        dic["ORGANO"] = dic["ORGANO"].replace('/', '_')
        dic["ORGANO"] = dic["ORGANO"].replace('"', '')
        
        output_folder_path = f"clean_data/{dic['LEGISLATURA']}/{(''.join(dic['ORGANO'].split(',')[0:2]))}/"
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        return output_folder_path
    
    
    def get_page_and_id(self, dic: dict) -> tuple[str | None, str | None]:

        pdf_id= re.search(r'[A-Z]+-\d+-[A-Z]+-\d+', dic["ENLACEPDF"])
        pdf_page = re.search(r'(?<=page=)\d*', dic["ENLACEPDF"])

        pdf_id = pdf_id.group() if pdf_id else None
        pdf_page = pdf_page.group() if pdf_page else None

        return pdf_id, pdf_page


    def get_output_files(self, dic: dict, text: str, intervention_num: int, interventions: dict, handled_pdfs: set, gender: str | None) -> None:

        if dic["LEGISLATURA"] == "Leg.15":
            dic["LEGISLATURA"] = "XV"
        
        output_folder_path = self.create_output_folder(dic)

        pdf_id, pdf_page = self.get_page_and_id(dic)

        if not pdf_id or not pdf_page:
            return

        if "ORADOR" in dic and text is not None and text.strip() not in ("No disponible", "") and dic["ORADOR"] != "Votación": 
            orator = re.sub(r'\(.*\)', "", dic["ORADOR"]) # corregir esta línea para que incluya caracteres como: à o ñ
            orator_code = re.findall(r'[A-Za-zÁÉÍÓÚÜÑÇáéíóúüñçÀÈÌÒÙàèìòù\'()]+', orator)
            party = re.search(r'\((.+)\)', dic["ORADOR"])
            party = party.group()[1:-1] if party else "No disponible"
            obj_i = re.sub(r'\s+', " ", dic.get("OBJETOINICIATIVA", "No disponble").strip())
            type_i = re.sub(r'\s+', " ", dic.get("TIPOINICIATIVA", "No disponble").strip())
            phase = re.sub(r'\s+', " ", dic.get("FASE", "No disponble").strip())
            role = re.sub(r'\s+', " ", dic.get("CARGOORADOR", "No disponble").strip())

            party, assigned = self.loader.assign_party(dic, party)

            id = f"{pdf_id}{pdf_page}{''.join(orator_code)}{intervention_num}"
            
            output_data = {"ID":id, "ORADOR":orator.strip(), "TEXTO":text, "CARGOORADOR":role or "No disponible", 
                            "PARTIDO":party, "LEGISLATURA":dic["LEGISLATURA"], "OBJETOINICIATIVA":obj_i or "No disponible",
                            "TIPOINICIATIVA":type_i or "No disponible", "ORGANO":dic["ORGANO"], "FASE":phase or "No disponible", 
                            "SESION":dic["SESION"], "INICIOINTERVENCION":dic.get("INICIOINTERVENCION") or "No disponible", 
                            "FININTERVENCION":dic.get("FININTERVENCION") or "No disponible", "IDPDF":pdf_id, "PAGPDF":pdf_page, 
                            "PARTIDOIMPUTADO":assigned, "GENEROIMPUTADO":gender}
          
            ruta_archivo_modificado = os.path.join(output_folder_path, f"Intervenciones_{pdf_id}.json")

            if pdf_id not in handled_pdfs:
                interventions[pdf_id] = []
                interventions[pdf_id].append(output_data)
                handled_pdfs.add(pdf_id)
            else:
                last_output_data = interventions[pdf_id][-1] if interventions[pdf_id] else None
                if output_data != last_output_data:
                    interventions[pdf_id].append(output_data)

            with open(ruta_archivo_modificado, 'w', encoding='utf-8') as f:                
                json.dump(interventions[pdf_id], f, ensure_ascii=False, indent=4)
            

def main(data: list, data_collector: DataCollector, n_workers: int, results_file_name: str, pdf_noorator: int) -> None:
    
    results, errors = [], []
    interventions, processed_interventions, special_orators = {}, {}, {}
    last_pdf = None
    distinct_leg = False
    handled_pdfs = set()
    intervention_num, pdf_nopage, errors_num, invalid_url_num, pdf_notext= 0, 0, 0, 0, 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        start = time.time()
        for dic in data:
            result = executor.submit(data_collector.process_intervention, dic, processed_interventions, special_orators, last_pdf, distinct_leg)
            result = result.result()
            results.append(result[0])
  
            pdf_nopage += result[1]
            errors_num += result[2]
            invalid_url_num += result[3]
            pdf_notext += result[4]
            errors.extend(result[5])
            processed_interventions.update(result[6])
            special_orators.update(result[7])
            last_pdf = result[8]
            gender = result[10]

            if result[9] == True:
                processed_interventions.clear()
                distinct_leg = False
  
            print(f"Intervencion{intervention_num+1} procesada")
            print(pdf_nopage, "|", errors_num, "|", invalid_url_num, "|", pdf_noorator, "|", pdf_notext)

            data_collector.get_output_files(dic, result[0], intervention_num, interventions, handled_pdfs, gender)

            intervention_num += 1

        end =  time.time()

    with open(results_file_name, "w", encoding="utf-8") as file:
        file.write(f"Número de intervenciones procesadas: {intervention_num}\n")
        file.write(f"Tiempo de procesamiento de las intervenciones: {round((end-start)/60, 4)} min | {round((end-start)/3600,4)} h\n")
        file.write(f"Número de intervenciones sin indicación de página en el enlace: {pdf_nopage}\n")
        file.write(f"Número de errores al intentar procesar las intervenciones: {errors_num}\n")
        file.write(f"Número de errores debido a la URL inválida del PDF: {invalid_url_num}\n")
        file.write(f"Número de intervenciones sin orador: {pdf_noorator}\n")
        file.write(f"Número de intervenciones sin texto disponible: {pdf_notext}\n")
        file.write("Lista de errores:\n")
        file.writelines(f"{error}\n" for error in errors)        

if __name__ == '__main__':
    
    folder_path = os.path.join('.', 'data/legislaturas')
    
    data_collector = DataCollector()
    data, pdf_noorator = data_collector.read_files(folder_path)

    loader = PartyLoader()
    loader.load_from_json_folder("data/organos_congreso")
    loader.load_from_csvs(["data/diputados/diputados11.csv", "data/diputados/diputados12.csv", "data/diputados/diputados13.csv", "data/diputados/diputados14.csv", "data/diputados/diputados15.csv", "data/senadores/senadores11.csv", "data/senadores/senadores12.csv", "data/senadores/senadores13.csv", "data/senadores/senadores14.csv", "data/senadores/senadores15.csv"])
    data_collector.loader = loader

    n_workers = 4
    results_file_name = "results/preprocessing/resultados_legislatura15.txt"

    main(data, data_collector, n_workers, results_file_name, pdf_noorator)
