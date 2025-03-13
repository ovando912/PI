#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include "/home/lucas/Documents/Proyecto_Integrador/KDSource/mcpl/src/mcpl/mcpl.h"

typedef struct TreeNode
{
    double *cumul;              // Puntero para el vector cumul
    double *micro;              // Puntero para el vector micro
    double *macro;              // Puntero para el vector macro
    int num_micro;              // Tamaño de micro
    int num_macro;              // Tamaño de macro
    struct TreeNode **children; // Hijos
    int num_children;           // Cantidad de hijos
} TreeNode;

// Crear un nodo nuevo
TreeNode *create_node()
{
    TreeNode *node = (TreeNode *)malloc(sizeof(TreeNode));
    if (!node)
    {
        fprintf(stderr, "Error al asignar memoria para el nodo\n");
        exit(EXIT_FAILURE);
    }
    node->cumul = NULL;
    node->micro = NULL;
    node->macro = NULL;
    node->num_micro = 0;
    node->num_macro = 0;
    node->children = NULL;
    node->num_children = 0;
    return node;
}

// Agregar un hijo a un nodo
void add_child(TreeNode *parent, TreeNode *child)
{
    parent->num_children++;
    parent->children = (TreeNode **)realloc(parent->children, parent->num_children * sizeof(TreeNode *));
    if (!parent->children)
    {
        fprintf(stderr, "Error al asignar memoria para los hijos\n");
        exit(EXIT_FAILURE);
    }
    parent->children[parent->num_children - 1] = child;
}

// Función para imprimir el árbol en consola
void print_tree(TreeNode *node, int level)
{
    for (int i = 0; i < level; i++)
        printf("  ");
    printf("Nodo:\n");

    // Imprimir los vectores
    printf("Cumul: ");
    if (node->num_micro > 0)
    {
        for (int i = 0; i < node->num_micro; i++)
        {
            printf("%.3lf ", node->cumul[i]);
        }
    }
    else
    {
        printf("None");
    }
    printf("\n");

    printf("Micro: ");
    if (node->num_micro > 0)
    {
        for (int i = 0; i < node->num_micro; i++)
        {
            printf("%.3lf ", node->micro[i]);
        }
    }
    else
    {
        printf("None");
    }
    printf("\n");

    printf("Macro: ");
    if (node->num_macro > 0)
    {
        for (int i = 0; i < node->num_macro; i++)
        {
            printf("%.3lf ", node->macro[i]);
        }
    }
    else
    {
        printf("None");
    }
    printf("\n");

    // Imprimir los hijos
    for (int i = 0; i < node->num_children; i++)
    {
        print_tree(node->children[i], level + 1);
    }
}

// Función para liberar la memoria del árbol
void free_tree(TreeNode *node)
{
    if (node == NULL)
        return;

    for (int i = 0; i < node->num_children; i++)
    {
        free_tree(node->children[i]);
    }
    if (node->children)
        free(node->children);
    if (node->cumul)
        free(node->cumul);
    if (node->micro)
        free(node->micro);
    if (node->macro)
        free(node->macro);
    free(node);
}

typedef struct
{
    double *geometria; // Arreglo de doubles para la geometría
    int geometria_len; // Número de elementos en geometria
    double z0;
    int N_original;
    char **fuente_original; // Arreglo de cadenas
    int fuente_original_len;
    char **columns_order; // Arreglo de cadenas
    int columns_order_len;
    int *micro_bins; // Arreglo de enteros
    int micro_bins_len;
    int *macro_bins; // Arreglo de enteros
    int macro_bins_len;
    char *binning_type;       // Cadena
    char *used_defined_edges; // Si lo dejas en una única cadena ya formateada
    double factor_normalizacion;
} Config;

void print_config(Config *config)
{
    if (config == NULL)
    {
        printf("La configuración es NULL.\n");
        return;
    }

    // Imprimir geometría (ahora como doubles)
    printf("Geometria: ");
    for (int i = 0; i < config->geometria_len; i++)
    {
        printf("%f", config->geometria[i]);
        if (i < config->geometria_len - 1)
            printf(", ");
    }
    printf("\n");

    // Imprimir z0 (como double)
    printf("z0: %f\n", config->z0);

    // Imprimir N_original (entero)
    printf("N_original: %d\n", config->N_original);

    // Imprimir fuente_original (arreglo de strings)
    printf("fuente_original: ");
    for (int i = 0; i < config->fuente_original_len; i++)
    {
        printf("%s", config->fuente_original[i]);
        if (i < config->fuente_original_len - 1)
            printf(", ");
    }
    printf("\n");

    // Imprimir columns_order (arreglo de strings)
    printf("columns_order: ");
    for (int i = 0; i < config->columns_order_len; i++)
    {
        printf("%s", config->columns_order[i]);
        if (i < config->columns_order_len - 1)
            printf(", ");
    }
    printf("\n");

    // Imprimir micro_bins (arreglo de enteros)
    printf("micro_bins: ");
    for (int i = 0; i < config->micro_bins_len; i++)
    {
        printf("%d", config->micro_bins[i]);
        if (i < config->micro_bins_len - 1)
            printf(", ");
    }
    printf("\n");

    // Imprimir macro_bins (arreglo de enteros)
    printf("macro_bins: ");
    for (int i = 0; i < config->macro_bins_len; i++)
    {
        printf("%d", config->macro_bins[i]);
        if (i < config->macro_bins_len - 1)
            printf(", ");
    }
    printf("\n");

    // Imprimir binning_type (cadena)
    printf("binning_type: %s\n", config->binning_type);

    // Imprimir used_defined_edges (cadena ya formateada)
    printf("used_defined_edges: %s\n", config->used_defined_edges);

    // Imprimir factor_normalizacion (double)
    printf("factor_normalizacion: %f\n", config->factor_normalizacion);
}

// Función auxiliar para recortar espacios
char *trim(char *str)
{
    while (*str == ' ' || *str == '\t' || *str == '\n')
        str++;
    if (*str == 0)
        return str;
    char *end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\t' || *end == '\n'))
    {
        *end = '\0';
        end--;
    }
    return str;
}

// Función para tokenizar una cadena y devolver un arreglo de cadenas
// Los tokens se separan por el delimitador dado (por ejemplo, ",")
char **tokenize_string(const char *str, const char *delim, int *count)
{
    char *temp = strdup(str);
    int capacity = 10;
    char **tokens = malloc(capacity * sizeof(char *));
    *count = 0;
    char *token = strtok(temp, delim);
    while (token != NULL)
    {
        if (*count >= capacity)
        {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(char *));
        }
        tokens[*count] = strdup(trim(token));
        (*count)++;
        token = strtok(NULL, delim);
    }
    free(temp);
    return tokens;
}

double *parse_double_array2(const char *str, int *count)
{
    int token_count = 0;
    char **tokens = tokenize_string(str, ",", &token_count);
    double *arr = malloc(token_count * sizeof(double));
    for (int i = 0; i < token_count; i++)
    {
        arr[i] = atof(tokens[i]); // Conversión a double
        free(tokens[i]);
    }
    free(tokens);
    *count = token_count;
    return arr;
}

// Ejemplo para extraer un arreglo de enteros desde una cadena
int *parse_int_array(const char *str, int *count)
{
    int token_count = 0;
    char **tokens = tokenize_string(str, ",", &token_count);
    int *arr = malloc(token_count * sizeof(int));
    for (int i = 0; i < token_count; i++)
    {
        arr[i] = atoi(tokens[i]);
        free(tokens[i]);
    }
    free(tokens);
    *count = token_count;
    return arr;
}

// Función para extraer un arreglo de strings desde una cadena
char **parse_string_array(const char *str, int *count)
{
    return tokenize_string(str, ",", count);
}

// Función para extraer la configuración desde el nodo <configuracion>
Config *parse_config(xmlNode *config_node)
{
    Config *cfg = malloc(sizeof(Config));
    // Inicializamos los punteros a NULL
    cfg->geometria = NULL;
    cfg->fuente_original = NULL;
    cfg->columns_order = NULL;
    cfg->micro_bins = NULL;
    cfg->macro_bins = NULL;
    cfg->binning_type = NULL;
    cfg->used_defined_edges = NULL;

    for (xmlNode *cur = config_node->children; cur; cur = cur->next)
    {
        if (cur->type != XML_ELEMENT_NODE)
            continue;
        char *content = (char *)xmlNodeGetContent(cur);
        if (strcmp((char *)cur->name, "geometria") == 0)
        {
            // Se espera contenido como "1, 15, 15, 100, 3, 3"
            cfg->geometria = parse_double_array2(content, &(cfg->geometria_len));
        }
        else if (strcmp((char *)cur->name, "z0") == 0)
        {
            cfg->z0 = atoi(content);
        }
        else if (strcmp((char *)cur->name, "N_original") == 0)
        {
            cfg->N_original = atoi(content);
        }
        else if (strcmp((char *)cur->name, "fuente_original") == 0)
        {
            cfg->fuente_original = parse_string_array(content, &(cfg->fuente_original_len));
        }
        else if (strcmp((char *)cur->name, "columns_order") == 0)
        {
            cfg->columns_order = parse_string_array(content, &(cfg->columns_order_len));
        }
        else if (strcmp((char *)cur->name, "micro_bins") == 0)
        {
            cfg->micro_bins = parse_int_array(content, &(cfg->micro_bins_len));
        }
        else if (strcmp((char *)cur->name, "macro_bins") == 0)
        {
            cfg->macro_bins = parse_int_array(content, &(cfg->macro_bins_len));
        }
        else if (strcmp((char *)cur->name, "binning_type") == 0)
        {
            cfg->binning_type = strdup(trim(content));
        }
        else if (strcmp((char *)cur->name, "used_defined_edges") == 0)
        {
            // En este caso, puedes guardarlo como cadena o procesarlo según lo necesites.
            cfg->used_defined_edges = strdup(trim(content));
        }
        else if (strcmp((char *)cur->name, "factor_normalizacion") == 0)
        {
            cfg->factor_normalizacion = atof(content);
        }
        xmlFree(content);
    }
    return cfg;
}

/**
 * Convierte una cadena de valores separados por comas a un arreglo de double.
 * Actualiza el entero 'count' con la cantidad de elementos encontrados.
 */
double *parse_double_array(const char *str, int *count)
{
    if (str == NULL)
    {
        *count = 0;
        return NULL;
    }

    // Se duplica la cadena para trabajar sin modificar el original
    char *s = strdup(str);
    if (!s)
        return NULL;

    // Se cuenta el número de valores (se asume que están separados por comas)
    int cnt = 0;
    for (char *p = s; *p; p++)
    {
        if (*p == ',')
            cnt++;
    }
    cnt++; // Cantidad de números = cantidad de comas + 1

    double *arr = (double *)malloc(cnt * sizeof(double));
    if (!arr)
    {
        free(s);
        *count = 0;
        return NULL;
    }

    int index = 0;
    char *token = strtok(s, ",");
    while (token != NULL)
    {
        arr[index++] = atof(token);
        token = strtok(NULL, ",");
    }
    *count = index;
    free(s);
    return arr;
}

/**
 * Función recursiva para parsear el XML y cargar la información en una estructura TreeNode.
 * Se asume que 'xml_node' es el puntero a la lista de nodos hijos de la etiqueta <node> actual.
 */
TreeNode *parse_xml_to_tree(xmlNode *xml_node)
{
    if (xml_node == NULL)
    {
        printf("Nodo XML vacío\n");
        return EXIT_FAILURE;
    }

    TreeNode *node = create_node();

    // Recorremos cada hijo del nodo XML actual
    for (xmlNode *cur = xml_node; cur != NULL; cur = cur->next)
    {
        if (cur->type != XML_ELEMENT_NODE)
            continue;

        if (xmlStrcmp(cur->name, (const xmlChar *)"cumul") == 0)
        {
            xmlChar *content = xmlNodeGetContent(cur);
            if (content && xmlStrcmp(content, (const xmlChar *)"None") == 0)
            {
                node->cumul = NULL;
            }
            else if (content)
            {
                node->cumul = parse_double_array((const char *)content, &node->num_micro);
            } // Aca guardo el tamaño de cumul en num_micro porque no quiero crear otra funcion
            // que no reciba ese parametro.
            else
                printf("Error al leer el contenido de la etiqueta 'cumul'\n");
            xmlFree(content);
        }
        else if (xmlStrcmp(cur->name, (const xmlChar *)"micro") == 0)
        {
            xmlChar *content = xmlNodeGetContent(cur);
            if (content && xmlStrcmp(content, (const xmlChar *)"None") == 0)
            {
                node->micro = NULL;
                node->num_micro = 0;
            }
            else if (content)
            {
                node->micro = parse_double_array((const char *)content, &node->num_micro);
            }
            else
                printf("Error al leer el contenido de la etiqueta 'micro'\n");
            xmlFree(content);
        }
        else if (xmlStrcmp(cur->name, (const xmlChar *)"macro") == 0)
        {
            xmlChar *content = xmlNodeGetContent(cur);
            if (content && xmlStrcmp(content, (const xmlChar *)"None") == 0)
            {
                node->macro = NULL;
                node->num_macro = 0;
            }
            else if (content)
            {
                node->macro = parse_double_array((const char *)content, &node->num_macro);
            }
            else
                printf("Error al leer el contenido de la etiqueta 'macro'\n");
            xmlFree(content);
        }
        else if (xmlStrcmp(cur->name, (const xmlChar *)"node") == 0)
        {
            // Procesamos recursivamente cada subnodo <node>
            TreeNode *childNode = parse_xml_to_tree(cur->children);
            if (childNode)
            {
                node->num_children++;
                node->children = realloc(node->children, node->num_children * sizeof(TreeNode *));
                if (!node->children)
                {
                    printf("Error al asignar memoria para los hijos\n");
                    return EXIT_FAILURE;
                }
                node->children[node->num_children - 1] = childNode;
            }
        }
    }

    // Según la lógica: si el vector 'macro' es "None", se supone que no hay hijos (ya se habrá guardado como NULL).
    // Además, si todos los vectores son None, permanecerán como NULL.
    return node;
}

// Función para cargar el árbol desde el XML
TreeNode *load_tree_from_xml(const char *filename, Config **out_config)
{
    xmlDoc *doc = xmlReadFile(filename, NULL, 0);
    if (!doc)
    {
        printf("Error al leer el archivo XML\n");
        return NULL;
    }

    xmlNode *root_element = xmlDocGetRootElement(doc);
    if (!root_element)
    {
        printf("El archivo XML está vacío o mal formado\n");
        xmlFreeDoc(doc);
        return NULL;
    }

    Config *cfg = NULL;
    TreeNode *tree_root = NULL;

    // Recorremos los hijos del nodo raíz
    for (xmlNode *cur = root_element->children; cur; cur = cur->next)
    {
        if (cur->type != XML_ELEMENT_NODE)
            continue;
        if (strcmp((char *)cur->name, "configuracion") == 0)
        {
            // Extraemos la configuración del primer nodo (o del que tenga este nombre)
            cfg = parse_config(cur);
        }
        else
        {
            // Aquí asumimos que los demás nodos son los de tu árbol y usas tu función parse_xml_to_tree
            tree_root = parse_xml_to_tree(cur);
        }
    }

    if (out_config != NULL)
        *out_config = cfg;

    xmlFreeDoc(doc);
    return tree_root;
}

// Función de interpolación lineal
double linear_interp(double y_target, double x[], double y[], int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        if (y_target >= y[i] && y_target <= y[i + 1])
        {
            // Fórmula de interpolación lineal: x = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
            return x[i] + (y_target - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]);
        }
    }
    if (size == 1)
    {
        return x[0];
    }
    printf("Valor de y_target fuera del rango\n");
    return -1; // Retorna -1 si y_target está fuera del rango
}

// Encuentra el índice del intervalo en el que se encuentra un valor
int find_interval(double value, double z[], int size_z)
{
    for (int i = 0; i < size_z - 1; i++)
    {
        if (value >= z[i] && value <= z[i + 1])
        {
            return i;
        }
    }
    printf("Valor de z fuera del rango\n");
    return -1; // Última posición posible
}

// Generacion de una particula
// Esta funcion no va a verificar que haya N niveles en el arbol para el array particle que ya tiene N posiciones guardadas
// De eso se debe encargar la funcion que este por arriba, la que samplee M particulas
void traverse(TreeNode *current, double *particle)
{
    int index = 0;
    while (current->num_children > 0)
    {
        double random_value = (double)rand() / RAND_MAX;
        double valor = linear_interp(random_value, current->micro, current->cumul, current->num_micro);
        int macrogrupo = find_interval(valor, current->macro, current->num_macro);
        current = current->children[macrogrupo];
        particle[index] = valor;
        index++;
    }
    double random_value = (double)rand() / RAND_MAX;
    double valor = linear_interp(random_value, current->micro, current->cumul, current->num_micro);
    particle[index] = valor;
}

void fill_particle(mcpl_particle_t *part, double *particle, double z0)
{
    part->ekin = 20 * exp(-particle[0]);

    part->polarisation[0] = 0;
    part->polarisation[1] = 0;
    part->polarisation[2] = 0;

    part->position[0] = particle[1];
    part->position[1] = particle[2];
    part->position[2] = z0;

    part->direction[0] = sin(acos(particle[3])) * cos(particle[4]);
    part->direction[1] = sin(acos(particle[3])) * sin(particle[4]);
    part->direction[2] = particle[3];

    part->time = 0.0;
    part->weight = 1.0;
    part->pdgcode = 2112;
    part->userflags = 0;
}

char *concatenate(const char *str1, const char *str2)
{
    // Calcula la longitud total de la cadena concatenada
    size_t len1 = strlen(str1);
    size_t len2 = strlen(str2);
    size_t total_len = len1 + len2 + 1; // +1 para el carácter nulo

    // Reserva memoria para la cadena concatenada
    char *result = (char *)malloc(total_len * sizeof(char));
    if (!result)
    {
        fprintf(stderr, "Error al asignar memoria\n");
        exit(EXIT_FAILURE);
    }

    // Copia la primera cadena en el resultado
    strcpy(result, str1);

    // Concatenar la segunda cadena
    strcat(result, str2);

    return result;
}

void display_usage()
{
    printf("Uso: programa -f <folder> -n <sample_count> [-s <source>] [-r <result>]\n");
    printf("   -f  Carpeta base (obligatoria)\n");
    printf("   -n  Cantidad de sampleos (obligatoria)\n");
    printf("   -s  Nombre del archivo source (opcional, por defecto \"source.xml\")\n");
    printf("   -r  Nombre del archivo result (opcional, por defecto \"sint_source.mcpl\")\n");
}

int parse_args(int argc, char **argv,
               const char **folder, const char **source,
               const char **result, long int *sample_count)
{
    // Valores por defecto
    *folder = NULL;
    *source = "source.xml";
    *result = "sint_source.mcpl";
    *sample_count = -1; // Valor no válido para detectar que no se asignó

    int i;
    for (i = 1; i < argc; i++)
    {
        // Ignorar argumentos vacíos
        if (argv[i][0] == '\0')
            continue;
        // Mostrar ayuda si se solicita
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            display_usage();
            exit(0);
        }
        // Opción para la carpeta
        if (strcmp(argv[i], "-f") == 0)
        {
            if (++i < argc)
            {
                *folder = argv[i];
                continue;
            }
            else
            {
                fprintf(stderr, "Error: Falta el valor para -f (folder).\n");
                exit(1);
            }
        }
        // Opción para la cantidad de sampleos
        if (strcmp(argv[i], "-n") == 0)
        {
            if (++i < argc)
            {
                *sample_count = atol(argv[i]);
                continue;
            }
            else
            {
                fprintf(stderr, "Error: Falta el valor para -n (sample_count).\n");
                exit(1);
            }
        }
        // Opción para el archivo source
        if (strcmp(argv[i], "-s") == 0)
        {
            if (++i < argc)
            {
                *source = argv[i];
                continue;
            }
            else
            {
                fprintf(stderr, "Error: Falta el valor para -s (source filename).\n");
                exit(1);
            }
        }
        // Opción para el archivo de salida
        if (strcmp(argv[i], "-r") == 0)
        {
            if (++i < argc)
            {
                *result = argv[i];
                continue;
            }
            else
            {
                fprintf(stderr, "Error: Falta el valor para -r (result filename).\n");
                exit(1);
            }
        }
        // Si el argumento empieza con '-' pero no coincide con ninguno válido
        if (argv[i][0] == '-')
        {
            fprintf(stderr, "Error: Argumento inválido: %s\n", argv[i]);
            exit(1);
        }
        // Si ya se asignaron, considerarlo un error por exceso de argumentos.
        fprintf(stderr, "Demasiados argumentos. Use -h para ayuda.\n");
        exit(1);
    }
    // Verificar que se hayan proporcionado los argumentos obligatorios
    if (!*folder)
    {
        fprintf(stderr, "Error: No se proporcionó la carpeta. Use -h para ayuda.\n");
        exit(1);
    }
    if (*sample_count == -1)
    {
        fprintf(stderr, "Error: No se proporcionó la cantidad de sampleos. Use -h para ayuda.\n");
        exit(1);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    printf("Inicio del programa\n");
    // Inicializar la semilla del generador de números aleatorios
    srand(time(NULL));

    const char *folder;
    const char *sourcename;
    const char *resultname;
    long int sample_count;

    parse_args(argc, argv, &folder, &sourcename, &resultname, &sample_count);

    // char *folder = "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/3-12-25/";
    // char *sourcename = "source.xml";
    // char *resultname = "sint_source.mcpl";

    char *sourcepath = concatenate(folder, sourcename);
    char *resultpath = concatenate(folder, resultname);

    mcpl_particle_t part;
    mcpl_outfile_t file = mcpl_create_outfile(resultpath);

    // Cargar el árbol desde el XML
    Config *config;
    TreeNode *root = load_tree_from_xml(sourcepath, &config);
    if (!root)
    {
        printf("Error al cargar el árbol desde el XML\n");
        return EXIT_FAILURE;
    }
    print_config(config);

    double *particle = (double *)malloc(config->columns_order_len * sizeof(double));
    if (!particle)
    {
        printf("Error al asignar memoria para la particula\n");
        return EXIT_FAILURE;
    }
    for (int j = 0; j < sample_count; j++)
    {
        traverse(root->children[0], particle);
        // for (int i = 0; i < particle_size; i++)
        // {
        //     printf("[%d]: %.3lf\t", i, particle[i]);
        // }
        // printf("\n");

        // Crear la partícula

        if ((j + 1) % 100000 == 0)
            printf("Particula %d\n", j + 1);

        fill_particle(&part, particle, 30);
        mcpl_add_particle(file, &part);
    }
    free(particle);

    // Liberar la memoria del árbol
    free_tree(root);
    mcpl_closeandgzip_outfile(file);
    printf("Fin del programa\n");

    return 0;
}
